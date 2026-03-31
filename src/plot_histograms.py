#################################################
# plot_histograms.py
# Lecture et visualisation des histogrammes
# sauvegardés par MC_Sampling.py
#################################################
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from SpectralFunction import *
from Constants import *

# Glück
def Glück_6He(r_rho):
    RC_gluck_6He = np.array([0.141, 0.138, 0.135, 0.132, 0.128, 0.121, 0.115, 0.108, 0.096, 0.082, 0.064, 0.041, 0.009, -0.039, -0.120, -0.336, -0.459, -0.599, -0.819, -1.209]) /100 + 1 - r_rho/100
    RC_gluck_abs_6He = np.array([99.4, 139.6, 182.4, 230.7, 285, 366.2, 433.5, 496.6, 596.2, 696, 795.5, 895, 994.5, 1093.2, 1187.8, 1293, 1320, 1340, 1360, 1380]) /1000

    return RC_gluck_abs_6He, RC_gluck_6He

def Glück_32Ar(r_rho):
    RC_gluck_32Ar = np.array([0.288, 0.297, 0.297, 0.289, 0.275, 0.267, 0.256, 0.243, 0.227, 0.206, 0.179, 0.143, 0.092, 0.015, -0.121, -0.214, -0.359, -0.622, -1.267, -2.156]) /100 + 1 - r_rho/100
    RC_gluck_abs_32Ar = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99]) /2

    return RC_gluck_abs_32Ar, RC_gluck_32Ar
#################################################
# Path to files
#################################################
FILES = {
    "electron"  : "hist/hist_electron_spectrum.txt",
    "ratio"     : "hist/hist_radiative_ratio.txt",
    "recoil"    : "hist/hist_recoil_spectrum.txt",
    "residual"  : "hist/hist_recoil_residual.txt",
}


def check_files():
    missing = [f for f in FILES.values() if not os.path.isfile(f)]
    if missing:
        print("Fichiers manquants :")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)


#################################################
# Chargement
#################################################
def load_all():
    data = {}
    for key, path in FILES.items():
        data[key] = np.loadtxt(path)
        print(f"[OK] {path}  →  {data[key].shape[0]} bins")
    return data


#################################################
# Figures
#################################################
def plot_electron_spectrum(d):
    """
    hist_electron_spectrum.txt
    Colonnes : E2_bin_center | hist_tree | hist_soft | hist_hard | hist_hard_sum
    """
    bins  = d[:, 0]
    tree  = d[:, 1]
    soft  = d[:, 2]
    hard  = d[:, 3]
    h_sum = d[:, 4]

    width = bins[1] - bins[0]

    fig, ax = plt.subplots()
    ax.bar(bins, tree,  width=width, label='Tree',      align='edge')
    ax.bar(bins, soft,  width=width, label='Soft',      align='edge')
    ax.bar(bins, hard,  width=width, label='Hard',      align='edge')
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title("Electron spectrum")
    ax.legend()
    fig.tight_layout()
    fig.savefig("hist/hist_electron_spectrum.pdf", dpi=300)
    return fig


def plot_radiative_ratio(d, electron_data, r_rho):
    """
    hist_radiative_ratio.txt
    Colonnes : E2_bin_center | ratio_distinguishable | ratio_indistinguishable | sirlin
    """
    bins_energy = electron_data[:, 0]

    E0 = float(bins_energy[-1])
    E2 = np.linspace(1.01, E0, 1000)
    Gardner = []
    for i in E2:
        Gardner.append(g_gardner(dE, i/me, CS, E0))

    bins        = d[:, 0]
    ratio_dist  = d[:, 1]
    ratio_indet = d[:, 2]
    sirlin      = d[:, 3]

    fig, ax = plt.subplots()
    ax.plot(bins, ratio_dist+r_rho/100,  color='tab:blue', label='Distinguishable photon')
    ax.plot(bins, ratio_indet+r_rho/100, color='tab:orange', label='Indistinguishable photon')
    ax.plot(bins, sirlin, color='tab:blue', linestyle='dashed', label='Sirlin')
    ax.plot(E2, Gardner, color='tab:orange', linestyle='dashed', label='Gardner')
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Radiative correction ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig("hist/hist_radiative_ratio.pdf", dpi=300)
    return fig


def plot_recoil_spectrum(d):
    """
    hist_recoil_spectrum.txt
    Colonnes : Er_bin_center | hist_tree | hist_soft | hist_hard
    """
    bins = d[:, 0]
    tree = d[:, 1]
    soft = d[:, 2]
    hard = d[:, 3]

    width = bins[1] - bins[0]

    fig, ax = plt.subplots()
    ax.bar(bins, tree, width=width, label='Tree', align='edge')
    ax.bar(bins, soft, width=width, label='Soft', align='edge')
    ax.bar(bins, hard, width=width, label='Hard', align='edge')
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title("Recoil spectrum")
    ax.legend()
    fig.tight_layout()
    fig.savefig("hist/hist_recoil_spectrum.pdf", dpi=300)
    return fig


def plot_recoil_residual(d, r_rho):
    """
    hist_recoil_residual.txt
    Colonnes : Er_bin_center | residual
    """
    bins     = d[:, 0]
    residual = d[:, 1]

    RC_gluck_abs_6He, RC_gluck_6He = Glück_32Ar(r_rho)


    fig, ax = plt.subplots()
    ax.plot(bins, residual, color='tab:blue', label='Recoil')
    ax.plot(RC_gluck_abs_6He, RC_gluck_6He, color='tab:orange', label='F.Glück 1998')
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig("hist/hist_recoil_residual.pdf", dpi=300)
    return fig


#################################################
# Main
#################################################
if __name__ == "__main__":

    r_rho = float(sys.argv[1])
    check_files()
    data = load_all()

    plot_electron_spectrum(data["electron"])
    plot_radiative_ratio(data["ratio"], data["electron"], r_rho)
    plot_recoil_spectrum(data["recoil"])
    plot_recoil_residual(data["residual"], r_rho)

    plt.show()
