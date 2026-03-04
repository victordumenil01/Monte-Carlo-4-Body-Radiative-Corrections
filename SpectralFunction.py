#################################################
# SpectralFunction.py file
#################################################
import numpy as np
from scipy.special import spence, gamma # type: ignore

from Constants import *
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