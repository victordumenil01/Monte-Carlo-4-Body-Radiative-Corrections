#################################################
# SpectralFunction.py file
#################################################
import numpy as np
from scipy.special import spence, gamma # type: ignore
from scipy import integrate # type: ignore
from scipy.interpolate import interp1d # type: ignore

from Constants import *

def end_point(mi, mf, betatype) :
    """
    Calculate the electron end point energy of the transition in units of me c^2
    
    :param mi: Atomic mass of the initial nucleus (in keV)
    :param mf: Atomic mass of the final nucleus (in keV)
    
    """
    delta = mi - mf
    e0 = 0
    if betatype == "beta-" :
        e0 = (delta + me)/me
    elif betatype == "beta+" :
        e0 = (delta - 1*me)/me
    Ecp_max = ( (((delta-me)/me)**2) + 2 * ((delta-me)/me))/(2*(mf/me)) # Maximum of kinetic energy of the recoil nucleus
    return e0 - Ecp_max

# fermi function
def fermi_function(W, Z, R):
    """Traditional Fermi Function

    :param Z: Proton number of the final nuclear state
    :param W: Electron energy in units of me c^2
    :param R: Nuclear radius in units of the electron Compton wavelength

    """
    f = 1

    if Z == 0:
        return f

    g = np.sqrt(1-(ALPHA*Z)**2)
    p = np.sqrt(W**2-1)
    y = ALPHA*Z*W/p

    #We use the traditional Fermi function, i.e. a prefactor 4 instead of 2(1+gamma)
    #This is consistent with the L0 description below

    f = (4
            *np.power(2*p*R, 2*(g-1))
            *np.exp(np.pi*y)
            *(np.abs(gamma(g+1.j*y))/(gamma(1+2*g)))**2)

    return f

# spence function
def L(x):
    return -1*spence(1-x)

# beta 
def beta_E(E):
    return ( 1 - me**2/(E**2) )**(1/2)

# z function
def zVS(E2, Delta, CS):
    E10 = Delta - E2
    
    beta = beta_E(E2)

    N = 0.5 * np.log( (1 + beta)/(1 - beta) )
    
    omega = CS * E10
    
    return (ALPHA/np.pi) * ( 1.5 * np.log(mp/me) 
                           + 2 * ( (N/beta) - 1 ) * np.log((2*omega)/me)
                           + (2 * N/beta) * (1 - N)
                           + (2/beta) * L(2 * beta/(1 + beta)) - (3/8) )

def zH(E2, Delta, CS):
    E10 = Delta - E2
    
    beta = beta_E(E2)

    N = 0.5 * np.log( (1 + beta)/(1 - beta) )
    
    omega = CS * E10
    
    return (ALPHA/np.pi) * (2*(N/beta-1)*(np.log(1/CS)+E10/(3*E2)-3/2)
                           + N/beta*(E10)**2/(12*E2**2))

# sirlin function
def sirlin_g(W, W0, **kwargs):
    """
    Sirlin's g function for order alpha radiative corrections

    :param W: Electron energy in units of me c^2
    :param W0: Electron endpoint energy in units of me c^2

    """
    g = 0

    beta = np.sqrt(W**2-1)/W

    g = (3*np.log(mp/me)
        -3./4.
        +4./beta*L((2*beta/(1+beta)))
        +4*(np.arctanh(beta)/beta-1)*((W0-W)/(3*W)-3/2+np.log(2*(W0-W)))
        +np.arctanh(beta)/beta*(2*(1+beta**2)+(W0-W)**2/(6*W**2)-4*np.arctanh(beta)))

    return g

# xi parameter
def xi_a(Lambda, MF, MGT):
    xi = MF**2 + (Lambda**2) * MGT**2 
    a = (MF**2 - (Lambda**2) * MGT**2/3)/xi

    return xi, a

# ─────────────────────────────────────────────
# Corrections radiatives de Gardner
# ─────────────────────────────────────────────

def _beta(eps):
    """Vitesse relativiste β = sqrt(1 - 1/ε²), vectorisée."""
    return np.sqrt(1.0 - 1.0 / eps**2)


def _I_hard(dE, E2, delta):
    """
    Intégrale I de Gardner pour la partie hard (photon dur, k > dE).
    dE    : coupure en énergie du photon (keV)
    E2    : énergie de l'électron (keV)
    delta : Q-value (keV)
    """
    b    = _beta(E2)
    kmax = delta - E2          # énergie maximale du photon
    dk   = kmax - dE           # largeur de l'intervalle

    # Polynômes en (kmax^n - dE^n) factorisés pour éviter les répétitions
    d1 = kmax - dE
    d2 = kmax**2 - dE**2
    d3 = kmax**3 - dE**3
    d4 = kmax**4 - dE**4

    E2sq  = E2**2
    kmaxsq = kmax**2

    ligne1 = (
        -4 * np.log(kmax / dE)
        - 4 * dk / E2
        + 8 * dk / kmax
    )
    ligne2 = (
          4 * d2 / (E2 * kmax - E2sq)
        - 2 * d2 / kmaxsq
        - (4/3) * d3 / (E2 * kmaxsq)
    )
    ligne3 = (
          2 * np.log(kmax / dE)
        + 2 * dk / E2
        + d2 / (2 * E2sq)
    )
    ligne4 = (
        - 4 * dk / kmax
        - 2 * d2 / (E2 * kmax - E2sq)
        - (2/3) * d3 / (E2sq * kmax)
    )
    ligne5 = (
          d2 / kmaxsq
        + (2/3) * d3 / (E2 * kmaxsq)
        + (1/4) * d4 / (E2sq * kmaxsq)
    )

    return ligne1 + ligne2 + (2.0 / b) * np.arctanh(b) * (ligne3 + ligne4 + ligne5)


def _Ik_m1(k, kmax, eps):
    """Intégrande I_k^{-1}(k) = log(...)/k."""
    be = _beta(eps)
    bk = _beta(eps - k)
    return np.log(((1 + bk) * (1 - be)) / ((1 + be) * (1 - bk))) / k


def _Ik_n(k, kmax, eps, n):
    """Intégrande I_k^n(k) = k^n * log((1+β_k)/(1-β_k))."""
    bk = _beta(eps - k)
    return k**n * np.log((1 + bk) / (1 - bk))


def _int_Ik(kmax, eps, n):
    """Intégrale de 0 à kmax de I_k^n (n entier >= 0)."""
    result, _ = integrate.quad(_Ik_n, 0, kmax, args=(kmax, eps, n))
    return result


def _int_Ik_m1(kmax, eps):
    """Intégrale de 0 à kmax de I_k^{-1}."""
    result, _ = integrate.quad(_Ik_m1, 0, kmax, args=(kmax, eps))
    return result


def _I_soft(kmax, eps, l):
    """
    Intégrale I_l de Gardner pour la partie soft (photon mou, k < kmax).
    kmax : énergie maximale du photon mou
    eps  : énergie de l'électron en unités de me
    l    : masse du photon infrarouge (régularisateur)
    """

    be = _beta(eps)

    # Logarithme de Spence : Li₂(x) avec la convention de scipy (spence(1-x) = Li₂(x))
    li2 = -spence(1.0 - 2.0 * be / (1.0 + be))

    ligne1 = (
          6.0
        - 4.0 * (1.0 - np.arctanh(be) / be) * np.log(2 * kmax / l)
        + 2.0 * np.arctanh(be) / be
        + (2.0 / be) * li2
    )
    ligne2 = (
        - 2.0 * np.arctanh(be)**2 / be
        + (2.0 / be)  * _int_Ik_m1(kmax, eps)
        - 2.0         * _int_Ik(kmax, eps, 0) / (eps * be)
        + 1.0         * _int_Ik(kmax, eps, 1) / (eps**2 * be)
        - 4.0 * np.sqrt((eps - kmax)**2 - 1.0) / np.sqrt(eps**2 - 1.0)
    )
    ligne3 = (4.0 / be) * np.log(
        (eps - kmax - np.sqrt((eps - kmax)**2 - 1.0))
        / (eps - np.sqrt(eps**2 - 1.0))
    )
    ligne4 = 4.0 * np.log(
        (eps**2 - 1.0 - eps * kmax + np.sqrt(eps**2 - 1.0) * np.sqrt((eps - kmax)**2 - 1.0))
        / (2.0 * (eps**2 - 1.0))
    )

    return ligne1 + ligne2 + ligne3 + ligne4


def _gb(dE, E2, l, delta):
    """
    Fonction gb de Gardner combinant partie soft et hard.
    Utilise la fonction de Heaviside pour sélectionner le bon régime.
    """
    # Partie soft : k_max = E2 - 1  (si E2 - 1 < dE, i.e. dE + 1 > E2)
    soft_low  = np.heaviside(-(dE + 1 - E2), 0) * _I_soft(E2 - 1,  E2, l)
    # Partie soft : k_max = dE       (si dE < E2 - 1)
    soft_high = np.heaviside(-(E2 - (dE + 1)), 0) * _I_soft(dE,     E2, l)
    # Partie hard
    hard      = np.heaviside(-(delta - (dE - E2)), 0) * _I_hard(dE, E2, delta)

    return soft_low + soft_high + hard


def _gvv(E2, l):
    """
    Correction virtuelle+soft de Gardner g_{vv}.
    E2 en unités de me, l masse IR du photon.
    """

    b  = _beta(E2)
    ath = np.arctanh(b)

    A = (
          0.5 * b * np.log((1 + b) / (1 - b))
        - 1.0
        + 2.0 * np.log(l) * (0.5 / b * np.log((1 + b) / (1 - b)) - 1.0)
        + 1.5 * np.log(mp / me)
        - (1.0 / b) * (0.5 * np.log((1 + b) / (1 - b)))**2
        + (1.0 / b) * (-spence(1.0 - 2.0 * b / (1.0 + b)))
    )
    return 2.0 * A - 0.75


def g_gardner(dE, E2, l, delta):
    """
    Facteur de correction radiative total de Gardner.
    g = 1 + (α/2π) * (g_b + g_vv) - 0.008

    Paramètres
    ----------
    dE    : coupure sur l'énergie du photon (keV)
    E2    : énergie de l'électron (keV, unités de me dans les sous-fonctions)
    l     : masse infrarouge du photon (régularisateur)
    delta : Q-value (keV)
    """
    return 1.0 + (ALPHA / (2.0 * np.pi)) * (_gb(dE, E2, l, delta) + _gvv(E2, l)) 


#─────────────────────────────────────────────
def build_spectrum_interpolator(file_path, Z, R):
    """
    Construit et retourne l'interpolateur de la fonction spectrale.
    À appeler UNE SEULE FOIS au démarrage, puis passer le callable
    retourné aux fonctions de sampling.

    Retourne
    --------
    f : callable
        f(E) évalue la fonction spectrale normalisée à l'énergie E
        (E en unités de me c^2). Vectorisée, retourne 0 hors domaine.
    """
    # --- Lecture ---
    data = np.loadtxt(file_path, skiprows=1)

    energy      = data[:, 0] / me + 1
    corrections = data[:, 1]
    C           = data[:, 2]

    # --- Quantités physiques ---
    E0 = max(energy)
    p  = np.sqrt(energy**2 - 1)

    weight = p * energy * (E0 - energy)**2 * fermi_function(energy, Z, R)
    g = 1 + (ALPHA / (2 * np.pi)) * sirlin_g(energy, E0)

    # --- Normalisation globale ---
    num = integrate.simpson(weight * C, x=energy)
    den = integrate.simpson(weight,     x=energy)

    C_norm  = C / (num / den)
    product = corrections * C_norm / g # divided by sirlin 

    # --- Interpolateur (construit une seule fois) ---
    interp_func = interp1d(
        energy, product,
        kind='cubic',
        bounds_error=False,
        fill_value=0.0
    )

    # Retourne directement le callable — pas d'évaluation ici
    return interp_func
