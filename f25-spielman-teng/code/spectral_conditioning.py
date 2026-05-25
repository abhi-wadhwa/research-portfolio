import numpy as np
from scipy.linalg import svdvals, hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)
RNG = np.random.default_rng(2024)

DIMS = [10, 20, 50, 100, 200]
SIGMA_SWEEP = np.logspace(-3, 0, 25)
N_TRIALS = 500


def condition_numbers(M, sigma, trials, rng):
    # kappa for M + sigma*G where G is gaussian
    n = M.shape[0]
    kappas = np.empty(trials)
    for t in range(trials):
        G = rng.standard_normal((n, n))
        svs = svdvals(M + sigma * G)
        kappas[t] = svs[0] / max(svs[-1], 1e-16)  # avoid div by 0
    return kappas


def plot_condition_scaling(save=True):
    # kappa vs sigma for zero matrix -- pure random case
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for n in DIMS:
        M = np.zeros((n, n))
        med_kappas = []
        for sigma in SIGMA_SWEEP:
            kappas = condition_numbers(M, sigma, N_TRIALS, RNG)
            med_kappas.append(np.median(kappas))
        ax.plot(SIGMA_SWEEP, med_kappas, "o-", markersize=3, label=f"n={n}")

    # n/sigma refs
    for n in [50, 200]:
        ref = n / SIGMA_SWEEP
        ax.plot(SIGMA_SWEEP, ref, "--", color="gray", alpha=0.4)
    ax.annotate(r"$n/\sigma$ reference", xy=(0.01, 5000), fontsize=9, color="gray")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Perturbation magnitude $\sigma$")
    ax.set_ylabel(r"Median $\kappa(\sigma G)$")
    ax.set_title("Condition Number Scaling: Pure Gaussian Matrices")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(FIGDIR, "condition_number_scaling.png")
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    plt.close(fig)


def plot_condition_histograms(save=True):
    # dist of log10(kappa) at fixed sigma
    sigma = 0.1
    fig, axes = plt.subplots(1, len(DIMS), figsize=(18, 3.5))

    for idx, n in enumerate(DIMS):
        M = np.zeros((n, n))
        kappas = condition_numbers(M, sigma, N_TRIALS, RNG)
        log_kappas = np.log10(kappas)
        ax = axes[idx]
        ax.hist(log_kappas, bins=40, density=True, alpha=0.7,
                color=plt.cm.plasma(idx / len(DIMS)))
        ax.set_title(f"n={n}")
        ax.set_xlabel(r"$\log_{10}\kappa$")
        if idx == 0:
            ax.set_ylabel("Density")

    fig.suptitle(
        rf"Distribution of $\log_{{10}}\kappa$ at $\sigma = {sigma}$",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save:
        path = os.path.join(FIGDIR, "condition_number_histograms.png")
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    plt.close(fig)


def plot_condition_vs_theory(save=True):
    # median kappa vs n/sigma for hilbert + zero base
    # TODO: check if this generalizes to non-square matrices
    fig, ax = plt.subplots(figsize=(8, 5.5))
    n = 50
    M_hilbert = hilbert(n)
    M_zero = np.zeros((n, n))

    for label, M in [("Hilbert", M_hilbert), ("Zero", M_zero)]:
        medians = []
        for sigma in SIGMA_SWEEP:
            kappas = condition_numbers(M, sigma, N_TRIALS, RNG)
            medians.append(np.median(kappas))
        ax.plot(SIGMA_SWEEP, medians, "o-", markersize=3, label=f"{label} (empirical)")

    # spielman-teng prediction
    theory = n / SIGMA_SWEEP
    ax.plot(SIGMA_SWEEP, theory, "k--", lw=2, alpha=0.6, label=r"$n/\sigma$ bound")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"Median $\kappa$")
    ax.set_title(f"Empirical Condition Number vs Spielman-Teng Bound (n={n})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(FIGDIR, "condition_vs_theory.png")
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    plt.close(fig)


def main():
    print("=== Spectral conditioning analysis ===")
    plot_condition_scaling()
    plot_condition_histograms()
    plot_condition_vs_theory()
    print("Done.\n")


if __name__ == "__main__":
    main()
