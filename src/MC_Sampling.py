#################################################
# MC_Sampling.py — version optimisée
#################################################
import numpy as np
from scipy import integrate # type: ignore
import os
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from Constants import *
from SpectralFunction import * # type: ignore

# ─────────────────────────────────────────────
# Fonctions de bas niveau (inchangées, vectorisées)
# ─────────────────────────────────────────────

def w0(E2, Delta, Lambda, MF, MGT, Z, R):
    """Densité spectrale à l'arbre (Eq. 5.19 CPC 101 223)."""
    E10 = Delta - E2
    beta = beta_E(E2)
    xi, _ = xi_a(Lambda, MF, MGT)
    return (Gv**2 * xi * beta * E10**2 * E2**2 * fermi_function(E2, Z, R)) / (2 * np.pi**3)


def wVS(E2, Delta, CS, Lambda, MF, MGT, Z, R):
    """Densité spectrale avec corrections virtuelles+soft (Eq. 5.20 CPC 101 223)."""
    beta = beta_E(E2)
    N = 0.5 * np.log((1 + beta) / (1 - beta))
    return w0(E2, Delta, Lambda, MF, MGT, Z, R) * (
        zVS(E2, Delta, CS) - (ALPHA * N / np.pi) * (1 - beta**2) / beta
    )


# ─────────────────────────────────────────────
# Intégrales rho (utilise np.linspace une seule fois)
# ─────────────────────────────────────────────

def _make_E2_grid(Delta, n=200):
    return np.linspace(me, Delta, n)


def rho0(Delta, Lambda, MF, MGT, Z, R):
    E2 = _make_E2_grid(Delta)
    return integrate.simpson(np.nan_to_num(w0(E2, Delta, Lambda, MF, MGT, Z, R)), x=E2)


def rhoVS(Delta, CS, Lambda, MF, MGT, Z, R):
    E2 = _make_E2_grid(Delta)
    return integrate.simpson(np.nan_to_num(wVS(E2, Delta, CS, Lambda, MF, MGT, Z, R)), x=E2)


# ─────────────────────────────────────────────
# Fonctions cinématiques — toutes vectorisées
# ─────────────────────────────────────────────

def P2(K, p2k, E2):
    return 1 / K**2 + me**2 / p2k**2 - 2 * E2 / K / p2k


def H0(E1, E2, K, P2_val, p2k):
    return E1 * (-(E2 + K) * P2_val + K / p2k)


def H1(E2, K, P2_val, p12, p1k, p2k):
    return p12 * (-P2_val + 1 / p2k) + p1k * ((E2 + K) / K / p2k - me**2 / p2k**2)


def g(beta, E2, p2k):
    N = 0.5 * np.log((1 + beta) / (1 - beta))
    return beta * E2 / (2 * N * p2k)


def MBR(E1, E2, K, p12, p1k, p2k, Lambda, MF, MGT, M, Z, R):
    xi, a = xi_a(Lambda, MF, MGT)
    P2_val = P2(K, p2k, E2)
    return (
        16 * Gv**2 * xi * M**2 * e**2
        * (H0(E1, E2, K, P2_val, p2k) + a * H1(E2, K, P2_val, p12, p1k, p2k))
        * fermi_function(E2 / me, Z, R)
    )


# ─────────────────────────────────────────────
# rho_H par Monte-Carlo (version vectorisée — un seul appel)
# ─────────────────────────────────────────────

def MC_rho_H(nH, Delta, CS, Lambda, MF, MGT, Z, R, M):
    """
    Calcule rho_H par intégration Monte-Carlo.
    Entièrement vectorisé : pas de boucle Python.
    """
    Vg = -32 * np.pi**3 * (Delta - me) * np.log(CS)

    rng = np.random.default_rng()          # générateur moderne, plus rapide
    u = rng.random((8, nH))
    u1, u2, u3, u4, u5, u6, u7, u8 = u

    E2   = me + (Delta - me) * u1
    E10  = Delta - E2
    omega = CS * E10
    K    = omega * np.exp(-u2 * np.log(CS))
    E1   = Delta - E2 - K

    mask = E1 > 0

    beta = beta_E(E2)
    N    = 0.5 * np.log((1 + beta) / (1 - beta))
    cg   = (1 - (1 + beta) * np.exp(-2 * N * u3)) / beta
    phig = 2 * np.pi * u6

    c1   = 2 * u4 - 1;  s1 = np.sqrt(1 - c1**2)
    c2   = 2 * u5 - 1;  s2 = np.sqrt(1 - c2**2)
    sg   = np.sqrt(1 - cg**2)

    phi1 = 2 * np.pi * u7
    phi2 = 2 * np.pi * u8

    n1 = np.stack((s1 * np.cos(phi1), s1 * np.sin(phi1), c1))          # (3, nH)

    n2     = np.stack((s2 * np.cos(phi2), s2 * np.sin(phi2), c2))
    n2pr   = np.stack((-np.sin(phi2),      np.cos(phi2),      np.zeros(nH)))
    n2prpr = np.stack((-c2 * np.cos(phi2), -c2 * np.sin(phi2), s2))

    n_perp = n2pr * np.cos(phig) + n2prpr * np.sin(phig)
    ng     = n2 * cg + n_perp * sg

    # produits scalaires via einsum (évite les boucles)
    p2k = E2 * K - beta * E2 * K * cg
    p1k = E1 * K * np.einsum('ij,ij->j', n1, ng)
    p12 = beta * E1 * E2 * np.einsum('ij,ij->j', n1, n2)

    g_calc  = g(beta, E2, p2k)
    mBR_val = MBR(E1, E2, K, p12, p1k, p2k, Lambda, MF, MGT, M, Z, R)

    weights = (K * beta * E1 * E2 * mBR_val / g_calc) / (2**13 * np.pi**8 * M**2)

    w_mask    = weights[mask]
    nH_corr   = np.count_nonzero(mask)
    w_max     = np.max(w_mask)

    rho_H = Vg / nH_corr * np.sum(w_mask)
    E_H   = np.sum(w_mask) / nH_corr / w_max

    return rho_H, E_H, w_max


# ─────────────────────────────────────────────
# Matrices d'amplitude
# ─────────────────────────────────────────────

def M0(E2, Delta, Lambda, MF, MGT, M, c, Z, R):
    E10  = Delta - E2
    beta = beta_E(E2)
    xi, a = xi_a(Lambda, MF, MGT)
    return (
        16 * Gv**2 * xi**2 * M**2 * E10 * E2
        * (1 + a * beta * c)
        * fermi_function(E2 / me, Z, R)
    )


def Mtilde(E2, Delta, Lambda, MF, MGT, M):
    E10  = Delta - E2
    beta = beta_E(E2)
    N    = 0.5 * np.log((1 + beta) / (1 - beta))
    xi, a = xi_a(Lambda, MF, MGT)
    return -ALPHA / np.pi * 16 * Gv**2 * (1 - beta**2) / beta * N * M**2 * E10 * E2 * xi


def MVS(E2, Delta, CS, Lambda, MF, MGT, M, c, Z, R):
    M_0     = M0(E2, Delta, Lambda, MF, MGT, M, c, Z, R)
    M_tilde = Mtilde(E2, Delta, Lambda, MF, MGT, M)
    z_VS    = zVS(E2, Delta, CS)
    return (z_VS * M_0 + M_tilde) * fermi_function(E2 / me, Z, R)


def W0(E2, Delta, Lambda, MF, MGT, M, c, Z, R):
    E10  = Delta - E2
    beta = beta_E(E2)
    M_0  = M0(E2, Delta, Lambda, MF, MGT, M, c, Z, R)
    return beta * E10 * E2 * M_0


def W0VS(E2, Delta, CS, Lambda, MF, MGT, M, c, Z, R):
    E10  = Delta - E2
    beta = beta_E(E2)
    M_0  = M0(E2, Delta, Lambda, MF, MGT, M, c, Z, R)
    M_VS = MVS(E2, Delta, CS, Lambda, MF, MGT, M, c, Z, R)
    return beta * E10 * E2 * (M_0 + M_VS)


# ─────────────────────────────────────────────
# Helpers : génération vectorielle de directions
# ─────────────────────────────────────────────

def _random_directions(n, c_col, phi_col=None):
    """
    Retourne n vecteurs unitaires 3D.
    c_col : cosinus polaires (n,)
    phi_col : angles azimutaux (n,) — tirés aléatoirement si None
    """
    if phi_col is None:
        phi_col = np.random.uniform(0, 2 * np.pi, n)
    s = np.sqrt(1 - c_col**2)
    return np.stack((s * np.cos(phi_col), s * np.sin(phi_col), c_col))  # (3, n)


def _build_frame(c2, phi2, n):
    """Construit la triade (n2, n2pr, n2prpr) à partir de c2, phi2."""
    s2     = np.sqrt(1 - c2**2)
    n2     = np.stack(( s2 * np.cos(phi2),  s2 * np.sin(phi2),  c2))
    n2pr   = np.stack((-np.sin(phi2),        np.cos(phi2),        np.zeros(n)))
    n2prpr = np.stack((-c2 * np.cos(phi2),  -c2 * np.sin(phi2),  s2))
    return n2, n2pr, n2prpr


# ─────────────────────────────────────────────
# Échantillonnage niveau-arbre
# ─────────────────────────────────────────────

def _sample_chunk(n_chunk, me, Delta, Lambda, MF, MGT, Z, R, M, w0_max):
    rng = np.random.default_rng()
    samples = []
    tries   = 0

    while len(samples) < n_chunk:
        tries += 1
        u1, u2, u3 = rng.random(3)
        e2 = me + (Delta - me) * u1
        c  = 2 * u2 - 1
        if u3 * w0_max <= W0(e2, Delta, Lambda, MF, MGT, M, c, Z, R):
            samples.append((e2, c))

    return np.array(samples), tries


def sampleTreeLevel(n, Delta, Lambda, MF, MGT, Z, R, M, num_threads):
    E2_grid = _make_E2_grid(Delta)
    _, a    = xi_a(Lambda, MF, MGT)
    sign    = 1 if a > 0 else -1
    w0_max  = np.nanmax(W0(E2_grid, Delta, Lambda, MF, MGT, M, sign, Z, R))

    chunk_sizes = [len(c) for c in np.array_split(np.arange(n), num_threads)]
    args = [(sz, me, Delta, Lambda, MF, MGT, Z, R, M, w0_max) for sz in chunk_sizes]

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        results = list(ex.map(lambda p: _sample_chunk(*p), args))

    E2_c_sampled = np.vstack([r[0] for r in results])[:n]
    total_tries  = sum(r[1] for r in results)
    print(f'[TreeLevel] Efficiency: {n / total_tries:.4f}')

    # Vecteurs de direction — entièrement vectorisés
    c2   = 2 * np.random.uniform(size=n) - 1
    phi2 = np.random.uniform(0, 2 * np.pi, n)
    phi1 = np.random.uniform(0, 2 * np.pi, n)

    n2, n2pr, n2prpr = _build_frame(c2, phi2, n)
    n_perp = n2pr * np.cos(phi1) + n2prpr * np.sin(phi1)
    c_vals = E2_c_sampled[:, 1]
    n1 = n2 * c_vals + n_perp * np.sqrt(1 - c_vals**2)

    return E2_c_sampled, n1, n2


# ─────────────────────────────────────────────
# Échantillonnage soft
# ─────────────────────────────────────────────

def _sample_chunk_soft(n_chunk, me, Delta, CS, Lambda, MF, MGT, Z, R, M, w0vs_max):
    rng     = np.random.default_rng()
    samples = []
    tries   = 0

    while len(samples) < n_chunk:
        tries += 1
        u1, u2, u3 = rng.random(3)
        e2 = me + (Delta - me) * u1
        c  = 2 * u2 - 1
        if u3 * w0vs_max <= W0VS(e2, Delta, CS, Lambda, MF, MGT, M, c, Z, R):
            samples.append((e2, c))

    return np.array(samples), tries


def sampleSoft(n, Delta, CS, Lambda, MF, MGT, Z, R, M, num_threads):
    E2_grid  = _make_E2_grid(Delta)
    _, a     = xi_a(Lambda, MF, MGT)
    sign     = 1 if a > 0 else -1
    w0vs_max = np.nanmax(W0VS(E2_grid, Delta, CS, Lambda, MF, MGT, M, sign, Z, R))

    chunk_sizes = [len(c) for c in np.array_split(np.arange(n), num_threads)]
    args = [(sz, me, Delta, CS, Lambda, MF, MGT, Z, R, M, w0vs_max) for sz in chunk_sizes]

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        results = list(ex.map(lambda p: _sample_chunk_soft(*p), args))

    E2_c_sampled = np.vstack([r[0] for r in results])[:n]
    total_tries  = sum(r[1] for r in results)
    print(f'[Soft] Efficiency: {n / total_tries:.4f}')

    c2   = 2 * np.random.uniform(size=n) - 1
    phi2 = np.random.uniform(0, 2 * np.pi, n)
    phi1 = np.random.uniform(0, 2 * np.pi, n)

    n2, n2pr, n2prpr = _build_frame(c2, phi2, n)
    n_perp = n2pr * np.cos(phi1) + n2prpr * np.sin(phi1)
    c_vals = E2_c_sampled[:, 1]
    n1 = n2 * c_vals + n_perp * np.sqrt(1 - c_vals**2)

    return E2_c_sampled, n1, n2


# ─────────────────────────────────────────────
# Décroissance à 3 corps
# ─────────────────────────────────────────────

def threeBodyDecay(fourMom1, m1, m2, m3, dir2, Q):
    """
    Calcule les quadri-impulsions des particules 2 et 3.
    Fonctionne sur des tableaux de shape (4, N) ou (3, N).
    """
    p1   = fourMom1[1:]                                    # (3, N)
    a    = m2 * m3
    b    = m3 * m3
    c    = np.linalg.norm(p1, axis=0)                     # (N,)
    d    = Q + m1 + m2 + m3 - fourMom1[0]
    e    = np.einsum('ij,ij->j', p1, dir2) / c            # cos entre p1 et dir2

    denom = 2 * (c**2 * e**2 - d**2)
    disc  = (
        a**2 * d**2 - 2 * a * b * d**2 + 4 * a * c**2 * d**2 * e**2
        - 2 * a * c**2 * d**2 - 2 * a * d**4 + b**2 * d**2
        + 2 * b * c**2 * d**2 - 2 * b * d**4 + c**4 * d**2
        - 2 * c**2 * d**4 + d**6
    )
    third  = a * c * e - b * c * e - c**3 * e + c * d**2 * e
    p2Norm = (-np.sqrt(disc) + third) / denom

    p2 = p2Norm * dir2
    p3 = -(p1 + p2)
    p3Norm = np.linalg.norm(p3, axis=0)

    fourMomentum2 = np.vstack((np.sqrt(a + p2Norm**2), p2))
    fourMomentum3 = np.vstack((np.sqrt(b + p3Norm**2), p3))

    return fourMomentum2, fourMomentum3


# ─────────────────────────────────────────────
# Échantillonnage hard (rejection sampling vectorisé)
# ─────────────────────────────────────────────

def _sample_chunk_hard(n_chunk, Delta, CS, Lambda, MF, MGT, Z, R, M, wmax,
                       batch_size=4096):
    """
    Génère n_chunk événements hard par rejection sampling par lots.
    batch_size : taille du batch vectorisé (tunable).
    """
    rng = np.random.default_rng()
    E_list, n1_list, n2_list, ng_list = [], [], [], []
    nSuccess = 0
    sum_w      = 0.0   # accumulateur pour <w> = (1/n) * sum(w_i)
    n_w        = 0     # nombre total de poids évalués (points valides E1>0)

    while nSuccess < n_chunk:
        u = rng.random((8, batch_size))
        u1, u2, u3, u4, u5, u6, u7, u8 = u

        E2   = me + (Delta - me) * u1
        E10  = Delta - E2
        omega = CS * E10
        K    = omega * np.exp(-u2 * np.log(CS))
        E1   = Delta - E2 - K

        valid = E1 > 0
        if not np.any(valid):
            continue

        # Restreint aux candidats valides
        E1_v = E1[valid]; E2_v = E2[valid]; K_v = K[valid]
        u3_v = u3[valid]; u4_v = u4[valid]; u5_v = u5[valid]
        u6_v = u6[valid]; u7_v = u7[valid]; u8_v = u8[valid]

        beta = beta_E(E2_v)
        N    = 0.5 * np.log((1 + beta) / (1 - beta))
        cg   = (1 - (1 + beta) * np.exp(-2 * N * u3_v)) / beta
        phig = 2 * np.pi * u6_v

        c1   = 2 * u4_v - 1;  s1 = np.sqrt(1 - c1**2)
        c2   = 2 * u5_v - 1;  s2 = np.sqrt(1 - c2**2)
        sg   = np.sqrt(1 - cg**2)

        phi1 = 2 * np.pi * u7_v
        phi2 = 2 * np.pi * u8_v

        n1 = np.stack((s1 * np.cos(phi1), s1 * np.sin(phi1), c1))
        n2, n2pr, n2prpr = _build_frame(c2, phi2, len(E2_v))
        n_perp = n2pr * np.cos(phig) + n2prpr * np.sin(phig)
        ng     = n2 * cg + n_perp * sg

        p2k = E2_v * K_v - beta * E2_v * K_v * cg
        p1k = E1_v * K_v * np.einsum('ij,ij->j', n1, ng)
        p12 = beta * E1_v * E2_v * np.einsum('ij,ij->j', n1, n2)

        g_calc  = g(beta, E2_v, p2k)
        mBR_val = MBR(E1_v, E2_v, K_v, p12, p1k, p2k, Lambda, MF, MGT, M, Z, R)
        weights = K_v * beta * E1_v * E2_v * mBR_val / g_calc / (2**13 * np.pi**8 * M**2)

        # Accumule les poids pour le calcul de l'efficacité
        sum_w += np.sum(weights)
        n_w   += len(weights)

        accept = rng.random(len(E2_v)) < weights / wmax
        n_acc  = np.count_nonzero(accept)
        if n_acc == 0:
            continue

        need = min(n_acc, n_chunk - nSuccess)
        idx  = np.where(accept)[0][:need]

        E_list.append(np.stack((E1_v[idx], E2_v[idx], K_v[idx]), axis=1))
        n1_list.append(n1[:, idx].T)
        n2_list.append(n2[:, idx].T)
        ng_list.append(ng[:, idx].T)
        nSuccess += need

    # E_H = 100 * <w> / w_max  avec <w> = (1/n) * sum(w_i)
    efficiency_H = 100.0 * (sum_w / n_w) / wmax

    return (
        np.vstack(E_list),
        np.vstack(n1_list),
        np.vstack(n2_list),
        np.vstack(ng_list),
        efficiency_H,
    )


def sampleHard(n, Delta, CS, Lambda, MF, MGT, Z, R, M, wmax, num_threads):
    chunk_sizes = [len(c) for c in np.array_split(np.arange(n), num_threads)]
    args = [(sz, Delta, CS, Lambda, MF, MGT, Z, R, M, wmax) for sz in chunk_sizes]

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        results = list(ex.map(lambda p: _sample_chunk_hard(*p), args))

    E_sampled  = np.vstack([r[0] for r in results])[:n]
    n1_sampled = np.vstack([r[1] for r in results])[:n]
    n2_sampled = np.vstack([r[2] for r in results])[:n]
    ng_sampled = np.vstack([r[3] for r in results])[:n]

    # Moyenne pondérée de l'efficacité sur tous les chunks
    eff_H = np.mean([r[4] for r in results])
    print(f'[Hard] Efficiency H: {eff_H:.4f} %')

    return E_sampled, n1_sampled, n2_sampled, ng_sampled


# ─────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────

def sampleEvents(A, Z, Delta, mi, MF, MGT, nTotal):
    R         = r0 * A**(1/3) / NATLENGTH
    M         = mi
    n_threads = os.cpu_count()

    Q      = Delta - me
    Er_max = (Q**2 + 2 * Q * me) / (2 * M)
    E0     = Delta

    # ── Niveau arbre ──
    E2_c_0, n1_0, n2_0 = sampleTreeLevel(nTotal, E0, Lambda, MF, MGT, Z, R, M, n_threads)

    pE_0        = np.sqrt(E2_c_0[:, 0]**2 - me**2) * n2_0
    fourMom2_0  = np.vstack((E2_c_0[:, 0], pE_0))
    fourMom1_0, fourMom3_0 = threeBodyDecay(fourMom2_0, me, 0, M, n1_0, Q)

    # ── Calcul des probabilités ──
    rho_H, _, wmax = MC_rho_H(nTotal, Delta, CS, Lambda, MF, MGT, Z, R, M)
    rho_0          = rho0(Delta, Lambda, MF, MGT, Z, R)
    rho_VS         = rhoVS(Delta, CS, Lambda, MF, MGT, Z, R)
    rho0VS         = rho_0 + rho_VS

    PH      = rho_H / (rho0VS + rho_H)
    r_rho   = 100 * (rho_VS + rho_H) / rho_0
    print(f"PH          : {PH:.6f}")
    print(f"r_rho       : {r_rho:.4f} %")

    nS = int(np.sum(np.random.uniform(size=nTotal) > PH))
    nH = nTotal - nS
    print(f"nS : {nS}   nH : {nH}   nS/nTotal : {nS / nTotal:.4f}")

    # ── Soft & Hard en parallèle ──
    with ProcessPoolExecutor() as ex:
        fut_S = ex.submit(sampleSoft, nS, E0, CS, Lambda, MF, MGT, Z, R, M, n_threads)
        fut_H = ex.submit(sampleHard, nH, Delta, CS, Lambda, MF, MGT, Z, R, M, wmax, n_threads)
        E2_c_S, n1_S, n2_S   = fut_S.result()
        E_H,    n1_H, n2_H, ng_H = fut_H.result()

    pE_S       = np.sqrt(E2_c_S[:, 0]**2 - me**2) * n2_S
    fourMom2_S = np.vstack((E2_c_S[:, 0], pE_S))
    fourMom1_S, fourMom3_S = threeBodyDecay(fourMom2_S, me, 0, M, n1_S, Q)

    # ── Cinématique hard ──
    p1_H = E_H[:, 0, None].T * n1_H.T          # (3, nH)
    p2_H = np.sqrt(E_H[:, 1]**2 - me**2)[:, None].T * n2_H.T
    pg_H = E_H[:, 2, None].T * ng_H.T
    pr_H = -(p1_H + p2_H + pg_H)
    Er   = np.linalg.norm(pr_H, axis=0)**2 / (2 * M)

    print("Done")

    # ── Figures ──
    bins   = np.linspace(me, Delta, 80)
    bins_r = np.linspace(0, Er_max, 80)

    plt.figure()
    plt.hist(E2_c_0[:, 0], bins=bins)
    plt.hist(E2_c_S[:, 0], bins=bins)
    plt.hist(E_H[:, 1], bins=bins)
    plt.xlabel("Energy (keV)")

    hist_0, _ = np.histogram(E2_c_0[:, 0], bins=bins)
    hist_S, _ = np.histogram(E2_c_S[:, 0], bins=bins)
    hist_H, _ = np.histogram(E_H[:, 1], bins=bins)
    hist_H_sum, _ = np.histogram(E_H[:, 1] + E_H[:, 2], bins=bins)

    plt.figure()
    plt.plot(bins[:-1], (hist_S+hist_H)/hist_0, label="Distinguishable photon")
    plt.plot(bins[:-1], (hist_S+hist_H_sum)/hist_0, label="Indistinguishable photon")
    plt.plot(bins[:-1], 1+ALPHA/(2*np.pi)*sirlin_g(bins[:-1]/me, Delta/me), label="Sirlin")
    plt.legend(loc=0)
    plt.xlabel("Energy (keV)")

    bins_r = np.linspace(0, Er_max, 50)

    hist_0_r, _ = np.histogram(fourMom3_0[0, :]-M, bins=bins_r)
    hist_S_r, _ = np.histogram(fourMom3_S[0, :]-M, bins=bins_r)
    hist_H_r, _ = np.histogram(Er, bins=bins_r)

    plt.figure()
    plt.hist(fourMom3_0[0, :]-M, bins=bins_r)
    plt.hist(fourMom3_S[0, :]-M, bins=bins_r)
    plt.hist(Er, bins=bins_r)
    plt.xlabel("Energy (keV)")


    plt.figure()
    plt.plot(bins_r[:-1], (hist_S_r + hist_H_r)/hist_0_r - r_rho)
    plt.xlabel("Energy (keV)")

     # ── Sauvegarde des histogrammes ──
    # Colonnes : centre_bin | valeurs...
    # Spectre électron (counts bruts)
    np.savetxt("hist/hist_electron_spectrum.txt",
               np.column_stack([bins[:-1], hist_0, hist_S, hist_H, hist_H_sum]),
               header="E2_bin_center  hist_tree  hist_soft  hist_hard  hist_hard_sum",
               fmt="%.6e")

    # Rapport aux corrections radiatives
    sirlin = 1 + ALPHA / (2 * np.pi) * sirlin_g(bins[:-1] / me, Delta / me)
    np.savetxt("hist/hist_radiative_ratio.txt",
               np.column_stack([bins[:-1],
                                (hist_S + hist_H)     / hist_0,
                                (hist_S + hist_H_sum) / hist_0,
                                sirlin]),
               header="E2_bin_center  ratio_distinguishable  ratio_indistinguishable  sirlin",
               fmt="%.6e")

    # Spectre de recul (counts bruts)
    np.savetxt("hist/hist_recoil_spectrum.txt",
               np.column_stack([bins_r[:-1], hist_0_r, hist_S_r, hist_H_r]),
               header="Er_bin_center  hist_tree  hist_soft  hist_hard",
               fmt="%.6e")

    # Résidu de recul
    np.savetxt("hist/hist_recoil_residual.txt",
               np.column_stack([bins_r[:-1], (hist_S_r + hist_H_r) / hist_0_r - r_rho]),
               header="Er_bin_center  residual",
               fmt="%.6e")

    print("Histogrammes sauvegardés dans :")
    print("  /hist/hist_electron_spectrum.txt")
    print("  /hist/hist_radiative_ratio.txt")
    print("  /hist/hist_recoil_spectrum.txt")
    print("  /hist/hist_recoil_residual.txt")

    plt.show()
