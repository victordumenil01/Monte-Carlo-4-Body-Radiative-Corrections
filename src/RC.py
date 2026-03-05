# Dependancies
import math
import numpy as np
from scipy import integrate
from scipy import optimize
from scipy.special import spence, gamma
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#Constantes 
me = m2 = 510.9989461 #keV, electron mass
mp = 938272.046 #keV, proton mass
mn = 939565.4133 #keV, neutron mass
ALPHA = 1/137 # fine structure constante
NATLENGTH = 386.15 # fm
e = (4 * np.pi * ALPHA)**(1/2)  #-1.6021766208 * 10**(-19), charge of electron
Gv = 1

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
    
    return (ALPHA/np.pi) * ( 1.5 * np.log(mp/m2) 
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
    """Sirlin's g function for order alpha radiative corrections

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

# w function
def w0(E2, Delta, Lambda, MF, MGT, Z, R): #Calculation of w0 with the simple integral
    # Eq. 5.19 in CPC 101 223
    E10 = Delta - E2
    
    beta = beta_E(E2)
    
    xi, _ = xi_a(Lambda, MF, MGT)
    
    return ( (Gv**2) * xi * beta * (E10**2) * (E2**2) * fermi_function(E2, Z, R) )/( 2 * (np.pi**3) )

def wVS(E2, Delta, CS, Lambda, MF, MGT, Z, R): #Calculation of wVS with the simple integral
    # Eq. 5.20 in CPC 101 223
    beta = beta_E(E2)
    
    N = 0.5 * np.log( (1 + beta)/(1 - beta) )
    
    return w0(E2, Delta, Lambda, MF, MGT, Z, R) * ( zVS(E2, Delta, CS) - ((ALPHA*N)/np.pi) * ((1 - (beta**2))/beta) )

# rho
def rho0(Delta, Lambda, MF, MGT, Z, R):
    E2 = np.linspace(me, Delta, 200)

    w = w0(E2, Delta, Lambda, MF, MGT, Z, R)

    return integrate.simpson(np.nan_to_num(w), x=E2)

def rhoVS(Delta, CS, Lambda, MF, MGT, Z, R):
    E2 = np.linspace(me, Delta, 200)

    w = wVS(E2, Delta, CS, Lambda, MF, MGT, Z, R)

    return integrate.simpson(np.nan_to_num(w), x=E2)

# 
def P2(K, p2k, E2):
    return 1/K**2+me**2/p2k**2-2*E2/K/p2k

def H0(E1, E2, K, P2, p2k):
    return E1*(-(E2+K)*P2+K/p2k)

def H1(E2, K, P2, p12, p1k, p2k):
    return p12*(-P2+1/p2k)+p1k*((E2+K)/K/p2k-me**2/p2k**2)

def g(beta, E2, p2k):
    N = 0.5 * np.log( (1 + beta)/(1 - beta) ) 
    return beta*E2/(2*N*p2k)

def MBR(E1, E2, K, p12, p1k, p2k, Lambda, MF, MGT, M, Z, R):
    xi, a = xi_a(Lambda, MF, MGT)
    
    P2_calc = P2(K, p2k, E2)
    
    return 16*Gv**2*xi*M**2*e**2*(H0(E1, E2, K, P2_calc, p2k)+a*H1(E2, K, P2_calc, p12, p1k, p2k))*fermi_function(E2/me, Z, R)

# 
def MC_rho_H(nH, Delta, CS, Lambda, MF, MGT, Z, R, M): #Calculation of rho_H
    Vg = -32 * (np.pi**3) * (Delta - m2) * np.log(CS)
    
    u1, u2, u3, u4, u5, u6, u7, u8 = np.random.uniform(size=(8, nH))
    
    E2 = me + (Delta-me)*u1
    E10 = Delta - E2
    omega = CS*E10
    K = omega*np.exp(-u2*np.log(CS))
    E1 = Delta - E2 - K

    mask = (E1 > 0)

    #print('Efficiency', np.sum(np.ones(nH)[mask])/nH)
    
    beta = beta_E(E2)
    
    N = 0.5 * np.log( (1 + beta)/(1 - beta) ) 
    
    cg = (1-(1+beta)*np.exp(-2*N*u3))/beta
    phig = 2*np.pi*u6
    
    c1 = 2*u4-1
    c2 = 2*u5-1
    phi1 = 2*np.pi*u7
    phi2 = 2*np.pi*u8
    
    s1 = np.sqrt(1-c1**2)
    s2 = np.sqrt(1-c2**2)
    sg = np.sqrt(1-cg**2)
    
    n1 = np.stack((s1*np.cos(phi1), s1*np.sin(phi1), c1), axis=-1).T
    
    n2 = np.stack((s2*np.cos(phi2), s2*np.sin(phi2), c2), axis=-1).T
    n2pr = np.stack((-np.sin(phi2), np.cos(phi2), np.zeros(nH)), axis=-1).T
    n2prpr = np.stack((-c2*np.cos(phi2), -c2*np.sin(phi2), s2), axis=-1).T
    
    n_perp = np.multiply(n2pr, np.cos(phig)) + np.multiply(n2prpr, np.sin(phig))
    
    ng = np.multiply(n2, cg)+np.multiply(n_perp, sg)
    
    p2k = E2*K-beta*E2*K*cg # four-product
    p1k = E1*K*(n1[0,:]*ng[0,:]+n1[1,:]*ng[1,:]+n1[2,:]*ng[2,:]) # three-product
    p12 = beta*E1*E2*(n1[0,:]*n2[0,:]+n1[1,:]*n2[1,:]+n1[2,:]*n2[2,:]) # three-product
    
    g_calc = g(beta, E2, p2k)

    mBR = MBR(E1, E2, K, p12, p1k, p2k, Lambda, MF, MGT, M, Z, R)

    weights = 1/(2**13*np.pi**8*M**2)*K*beta*E1*E2*mBR/g_calc

    nH_corr = np.sum(np.ones(nH)[mask])

    rho_H = Vg/nH_corr*np.sum(weights[mask])

    E_H = np.sum(weights[mask])/nH_corr/np.max(weights[mask])

    return rho_H, E_H, np.max(weights[mask])

# 
def M0(E2, Delta, Lambda, MF, MGT, M, c, Z, R):
    E10 = Delta-E2

    beta = beta_E(E2)
    
    xi, a = xi_a(Lambda, MF, MGT)

    return 16*Gv**2*xi**2*M**2*E10*E2*(1+a*beta*c)*fermi_function(E2/me, Z, R)

def Mtilde(E2, Delta, Lambda, MF, MGT, M):
    E10 = Delta-E2

    beta = beta_E(E2)

    N = 0.5*np.log((1+beta)/(1-beta))
    
    xi, a = xi_a(Lambda, MF, MGT)

    return -ALPHA/np.pi*16*Gv**2*(1-beta**2)/beta*N*M**2*E10*E2*xi

def MVS(E2, Delta, CS, Lambda, MF, MGT, M, c, Z, R):
    M_0 = M0(E2, Delta, Lambda, MF, MGT, M, c, Z, R)
    M_tilde = Mtilde(E2, Delta, Lambda, MF, MGT, M)

    z_VS = zVS(E2, Delta, CS)

    return (z_VS*M_0 + M_tilde)*fermi_function(E2/me, Z, R)

def W0(E2, Delta, Lambda, MF, MGT, M, c, Z, R):
    E10 = Delta-E2

    beta = beta_E(E2)

    M_0 = M0(E2, Delta, Lambda, MF, MGT, M, c, Z, R)

    return beta*E10*E2*M_0

def W0VS(E2, Delta, CS, Lambda, MF, MGT, M, c, Z, R):
    E10 = Delta-E2

    beta = beta_E(E2)

    M_0 = M0(E2, Delta, Lambda, MF, MGT, M, c, Z, R)

    M_VS = MVS(E2, Delta, CS, Lambda, MF, MGT, M, c, Z, R)

    return beta*E10*E2*(M_0+M_VS)

# 3 sample
def _sample_chunk(n_chunk, me, Delta, Lambda, MF, MGT, Z, R, M, w0_max):
    """Génère un bloc d'échantillons acceptés."""
    samples = []
    tries = 0
    nSuccess = 0

    while nSuccess < n_chunk:
        tries += 1
        u1, u2, u3 = np.random.uniform(size=3)
        e2 = me + (Delta - me) * u1
        c = 2 * (u2 - 0.5)
        f = u3 * w0_max

        if f <= W0(e2, Delta, Lambda, MF, MGT, M, c, Z, R):
            samples.append([e2, c])
            nSuccess += 1

    return np.array(samples), tries

def sampleTreeLevel(n, Delta, Lambda, MF, MGT, Z, R, M, num_threads):
    # Pré-calculs
    E2 = np.linspace(me, Delta, 200)
    _, a = xi_a(Lambda, MF, MGT)
    w0 = np.nan_to_num(W0(E2, Delta, Lambda, MF, MGT, M, 1 if a > 0 else -1, Z, R))
    w0_max = np.max(w0)

    # Répartition équitable des tâches
    n_split = np.array_split(np.arange(n), num_threads)
    args = [(len(idx), me, Delta, Lambda, MF, MGT, Z, R, M, w0_max) for idx in n_split]

    # Thread pool
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(lambda p: _sample_chunk(*p), args))

    # Rassembler les résultats
    E2_c_list = [res[0] for res in results]
    tries_list = [res[1] for res in results]

    E2_c_sampled = np.vstack(E2_c_list)[:n]
    total_tries = sum(tries_list)

    print('Efficiency:', n / total_tries)

    # Calcul des vecteurs (déjà bien vectorisé)
    c2 = 2 * np.random.uniform(size=n) - 1
    s2 = np.sqrt(1 - c2**2)
    phi2 = np.random.uniform(size=n) * 2 * np.pi
    phi1 = np.random.uniform(size=n) * 2 * np.pi

    n2 = np.stack((s2 * np.cos(phi2), s2 * np.sin(phi2), c2), axis=-1).T
    n2pr = np.stack((-np.sin(phi2), np.cos(phi2), np.zeros(n)), axis=-1).T
    n2prpr = np.stack((-c2 * np.cos(phi2), -c2 * np.sin(phi2), s2), axis=-1).T

    n_perp = n2pr * np.cos(phi1) + n2prpr * np.sin(phi1)
    n1 = n2 * E2_c_sampled[:, 1] + n_perp * np.sqrt(1 - E2_c_sampled[:, 1]**2)

    return E2_c_sampled, n1, n2

def _sample_chunk_soft(n_chunk, me, Delta, CS, Lambda, MF, MGT, Z, R, M, w0vs_max):
    """Génère un bloc d'échantillons acceptés pour sampleSoft."""
    samples = []
    tries = 0
    nSuccess = 0

    while nSuccess < n_chunk:
        tries += 1
        u1, u2, u3 = np.random.uniform(size=3)
        e2 = me + (Delta - me) * u1
        c = 2 * (u2 - 0.5)
        f = u3 * w0vs_max

        if f <= W0VS(e2, Delta, CS, Lambda, MF, MGT, M, c, Z, R):
            samples.append([e2, c])
            nSuccess += 1

    return np.array(samples), tries


def sampleSoft(n, Delta, CS, Lambda, MF, MGT, Z, R, M, num_threads):
    # Pré-calculs
    E2 = np.linspace(me, Delta, 200)
    _, a = xi_a(Lambda, MF, MGT)
    w0vs = np.nan_to_num(W0VS(E2, Delta, CS, Lambda, MF, MGT, M, 1 if a > 0 else -1, Z, R))
    w0vs_max = np.max(w0vs)

    # Distribution équitable
    n_split = np.array_split(np.arange(n), num_threads)
    args = [(len(idx), me, Delta, CS, Lambda, MF, MGT, Z, R, M, w0vs_max) for idx in n_split]

    # Thread pool
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(lambda p: _sample_chunk_soft(*p), args))

    # Rassembler les échantillons
    E2_c_list = [res[0] for res in results]
    tries_list = [res[1] for res in results]

    E2_c_sampled = np.vstack(E2_c_list)[:n]
    total_tries = sum(tries_list)

    print('Efficiency:', n / total_tries)

    # Vecteurs aléatoires
    c2 = 2 * np.random.uniform(size=n) - 1
    s2 = np.sqrt(1 - c2**2)
    phi2 = np.random.uniform(size=n) * 2 * np.pi
    phi1 = np.random.uniform(size=n) * 2 * np.pi

    n2 = np.stack((s2 * np.cos(phi2), s2 * np.sin(phi2), c2), axis=-1).T
    n2pr = np.stack((-np.sin(phi2), np.cos(phi2), np.zeros(n)), axis=-1).T
    n2prpr = np.stack((-c2 * np.cos(phi2), -c2 * np.sin(phi2), s2), axis=-1).T

    n_perp = n2pr * np.cos(phi1) + n2prpr * np.sin(phi1)
    n1 = n2 * E2_c_sampled[:, 1] + n_perp * np.sqrt(1 - E2_c_sampled[:, 1]**2)

    return E2_c_sampled, n1, n2

#
def threeBodyDecay(fourMom1, m1, m2, m3, dir2, Q):
    p1 = fourMom1[1:]
    a = m2*m3
    b = m3*m3
    c = np.linalg.norm(p1, axis=0)
    d = Q + m1 + m2 + m3 - fourMom1[0]
    e = (p1[0]*dir2[0]+p1[1]*dir2[1]+p1[2]*dir2[2])/c

    first = 1./2./(c*c*e*e-d*d)
    second = (a*a*d*d-2*a*b*d*d+4.*a*c*c*d*d*e*e
              -2.*a*c*c*d*d-2.*a*d*d*d*d+b*b*d*d
              +2*b*c*c*d*d-2.*b*d*d*d*d+c*c*c*c*d*d
              -2.*c*c*d*d*d*d+d*d*d*d*d*d)
    third = a*c*e-b*c*e-c*c*c*e+c*d*d*e

    p2Norm = first*(-np.sqrt(second)+third)
    p2 = p2Norm*dir2
    p3 = -(p1+p2)
    
    p3Norm = np.linalg.norm(p3, axis=0)

    fourMomentum2 = np.array([np.sqrt(a+p2Norm**2), *p2])
    fourMomentum3 = np.array([np.sqrt(b+p3Norm**2), *p3])

    return fourMomentum2, fourMomentum3


def _sample_chunk_hard(n_chunk, Delta, CS, Lambda, MF, MGT, Z, R, M, wmax):
    """Génère un bloc d'échantillons acceptés pour sampleHard."""
    E_list, n1_list, n2_list, ng_list = [], [], [], []
    nSuccess = 0

    while nSuccess < n_chunk:
        u = np.random.uniform(size=8)
        u1, u2, u3, u4, u5, u6, u7, u8 = u
        
        E2 = me + (Delta - me) * u1
        E10 = Delta - E2
        omega = CS * E10
        K = omega * np.exp(-u2 * np.log(CS))
        E1 = Delta - E2 - K

        if E1 <= 0:
            continue

        beta = beta_E(E2)
        N = 0.5 * np.log((1 + beta)/(1 - beta))
        cg = (1 - (1 + beta) * np.exp(-2 * N * u3)) / beta
        phig = 2 * np.pi * u6
        
        c1 = 2 * u4 - 1
        c2 = 2 * u5 - 1
        phi1 = 2 * np.pi * u7
        phi2 = 2 * np.pi * u8
        
        s1 = np.sqrt(1 - c1**2)
        s2 = np.sqrt(1 - c2**2)
        sg = np.sqrt(1 - cg**2)
        
        n1 = np.array([s1*np.cos(phi1), s1*np.sin(phi1), c1])
        n2 = np.array([s2*np.cos(phi2), s2*np.sin(phi2), c2])
        n2pr = np.array([-np.sin(phi2), np.cos(phi2), 0])
        n2prpr = np.array([-c2*np.cos(phi2), -c2*np.sin(phi2), s2])
        n_perp = n2pr * np.cos(phig) + n2prpr * np.sin(phig)
        ng = n2 * cg + n_perp * sg
        
        p2k = E2 * K - beta * E2 * K * cg
        p1k = E1 * K * np.dot(n1, ng)
        p12 = beta * E1 * E2 * np.dot(n1, n2)
        
        g_calc = g(beta, E2, p2k)
        mBR_val = MBR(E1, E2, K, p12, p1k, p2k, Lambda, MF, MGT, M, Z, R)
        weight = 1 / (2**13 * np.pi**8 * M**2) * K * beta * E1 * E2 * mBR_val / g_calc
        
        if np.random.uniform() < weight / wmax:
            E_list.append([E1, E2, K])
            n1_list.append(n1)
            n2_list.append(n2)
            ng_list.append(ng)
            nSuccess += 1

    return (np.array(E_list), np.array(n1_list), np.array(n2_list), np.array(ng_list))


def sampleHard(n, Delta, CS, Lambda, MF, MGT, Z, R, M, wmax, num_threads):
    # Répartition équitable
    n_split = np.array_split(np.arange(n), num_threads)
    args = [(len(idx), Delta, CS, Lambda, MF, MGT, Z, R, M, wmax) for idx in n_split]

    # Thread pool
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(lambda p: _sample_chunk_hard(*p), args))

    # Rassembler les résultats
    E_sampled   = np.vstack([res[0] for res in results])[:n]
    n1_sampled  = np.vstack([res[1] for res in results])[:n]
    n2_sampled  = np.vstack([res[2] for res in results])[:n]
    ng_sampled  = np.vstack([res[3] for res in results])[:n]

    return E_sampled, n1_sampled, n2_sampled, ng_sampled

def sampleEvents(A, Z, Delta, MF, MGT, nTotal):
    #Delta = 3505+me
    #MF = 0
    #MGT = 6
    #Z = 2
    #A = 6
    #nTotal = int(1e5)
    Lambda = 1.2754
    N = (A-Z)
    R = 1.2*A**0.33/NATLENGTH
    CS = 0.001
    M = (Z*mp + N*mn)
    n_threads = os.cpu_count()
    

    Q = Delta-me

    Er_max = (Q**2+2*Q*me)/2/M

    print(Er_max)

    E0 = Delta-Er_max

    

    # Tree Level

    E2_c_0, n1_0, n2_0 = sampleTreeLevel(nTotal, E0, Lambda, MF, MGT, Z, R, M, n_threads)

    pE_0 = np.sqrt(E2_c_0[:, 0]**2-me**2)*n2_0

    fourMom2_0 = np.stack((E2_c_0[:, 0], pE_0[0, :], pE_0[1, :], pE_0[2, :]))

    fourMom1_0, fourMom3_0 = threeBodyDecay(fourMom2_0, me, 0, M, n1_0, Q)

    # Calculate P_H

    rho_H, _, wmax = MC_rho_H(nTotal, Delta, CS, Lambda, MF, MGT, Z, R, M)

    rho_0 = rho0(Delta, Lambda, MF, MGT, Z, R)

    rho_VS = rhoVS(Delta, CS, Lambda, MF, MGT, Z, R)

    rho0VS = rho_0+rho_VS

    PH = rho_H/(rho0VS+rho_H)
    print("PH :", PH)

    r_rho = 100*(rho_VS+rho_H)/rho_0
    print("r_rho :", r_rho)
    # Determine number soft/hard

    nS = np.sum(np.random.uniform(size=nTotal) > PH)
    nH = nTotal-nS

    print(nS, nH)
    print(nS/nTotal)

    # Sample soft 
    
    with ProcessPoolExecutor() as executor:
        future_soft = executor.submit(sampleSoft, nS, E0, CS, Lambda, MF, MGT, Z, R, M, n_threads)
        future_hard = executor.submit(sampleHard, nH, Delta, CS, Lambda, MF, MGT, Z, R, M, wmax, n_threads)

        E2_c_S, n1_S, n2_S = future_soft.result()
        E_H, n1_H, n2_H, ng_H = future_hard.result()
    

    pE_S = np.sqrt(E2_c_S[:, 0]**2-me**2)*n2_S

    fourMom2_S = np.stack((E2_c_S[:, 0], pE_S[0, :], pE_S[1, :], pE_S[2, :]))

    fourMom1_S, fourMom3_S = threeBodyDecay(fourMom2_S, me, 0, M, n1_S, Q)

    # sample hard
    p1_H = E_H[:, 0]*n1_H.T
    p2_H = np.sqrt(E_H[:, 1]**2-me**2)*n2_H.T
    pg_H = E_H[:, 2]*ng_H.T

    pr_H = -(p1_H+p2_H+pg_H)

    Er = np.linalg.norm(pr_H, axis=0)**2/2/M

    print("Done")

    bins = np.linspace(me, Delta, 50)

    plt.figure()
    plt.hist(E2_c_0[:, 0], bins=bins)
    plt.hist(E2_c_S[:, 0], bins=bins)
    plt.hist(E_H[:, 1], bins=bins)

    hist_0, _ = np.histogram(E2_c_0[:, 0], bins=bins)
    hist_S, _ = np.histogram(E2_c_S[:, 0], bins=bins)
    hist_H, _ = np.histogram(E_H[:, 1], bins=bins)
    hist_H_sum, _ = np.histogram(E_H[:, 1] + E_H[:, 2], bins=bins)

    plt.figure()
    plt.plot(bins[:-1], (hist_S+hist_H)/hist_0, label="Distinguishable photon")
    plt.plot(bins[:-1], (hist_S+hist_H_sum)/hist_0, label="Indistinguishable photon")
    plt.plot(bins[:-1], 1+ALPHA/(2*np.pi)*sirlin_g(bins[:-1]/me, Delta/me), label="Sirlin")
    plt.legend(loc=0)

    bins_r = np.linspace(0, 1.5, 50)

    hist_0_r, _ = np.histogram(fourMom3_0[0, :]-M, bins=bins_r)
    hist_S_r, _ = np.histogram(fourMom3_S[0, :]-M, bins=bins_r)
    hist_H_r, _ = np.histogram(Er, bins=bins_r)

    plt.figure()
    plt.hist(fourMom3_0[0, :]-M, bins=bins_r)
    plt.hist(fourMom3_S[0, :]-M, bins=bins_r)
    plt.hist(Er, bins=bins_r)

    r_rho_C = (rho_VS+rho_H)/rho_0

    print(r_rho_C)

    plt.figure()
    plt.plot(bins_r[:-1], (hist_S_r + hist_H_r)/hist_0_r - r_rho_C)
    plt.show()

    
if __name__ == "__main__":
    A = int(input("Enter mass number A: "))
    Z = int(input("Enter atomic number Z: "))
    Delta = float(input("Enter energy difference Delta (in keV): "))
    MF = float(input("Enter Fermi matrix element MF: "))
    MGT = int(input("Enter Gamow-Teller matrix element MGT: "))
    nTotal = int(input("Enter number of events nTotal: "))
    sampleEvents(A, Z, Delta, MF, MGT, nTotal)
