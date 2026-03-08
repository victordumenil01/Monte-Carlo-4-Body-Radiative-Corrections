#################################################
# plot_histograms.py
# Lecture et visualisation des histogrammes
# sauvegardés par MC_Sampling.py
#################################################
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

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
    return fig


def plot_radiative_ratio(d):
    """
    hist_radiative_ratio.txt
    Colonnes : E2_bin_center | ratio_distinguishable | ratio_indistinguishable | sirlin
    """
    bins        = d[:, 0]
    ratio_dist  = d[:, 1]
    ratio_indet = d[:, 2]
    sirlin      = d[:, 3]

    fig, ax = plt.subplots()
    ax.plot(bins, ratio_dist,  label='Photon détectable')
    ax.plot(bins, ratio_indet, label='Photon non détectable')
    ax.plot(bins, sirlin,      label='Sirlin')
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Ratio")
    ax.legend()
    fig.tight_layout()
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
    return fig


def plot_recoil_residual(d):
    """
    hist_recoil_residual.txt
    Colonnes : Er_bin_center | residual
    """
    bins     = d[:, 0]
    residual = d[:, 1]

    fig, ax = plt.subplots()
    ax.plot(bins, residual)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Ratio")
    fig.tight_layout()
    return fig


#################################################
# Main
#################################################
if __name__ == "__main__":
    check_files()
    data = load_all()

    plot_electron_spectrum(data["electron"])
    plot_radiative_ratio(data["ratio"])
    plot_recoil_spectrum(data["recoil"])
    plot_recoil_residual(data["residual"])

    plt.show()
