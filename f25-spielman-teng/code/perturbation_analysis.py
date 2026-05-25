import numpy as np
from scipy.linalg import svdvals
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)
RNG = np.random.default_rng(123)

N_TRIALS = 1500


def compute_least_svs(n, trials, rng, dist="gaussian"):
    out = np.empty(trials)
    for i in range(trials):
        if dist == "gaussian":
            A = rng.standard_normal((n, n)) / np.sqrt(n)
        else:
            A = rng.choice([-1.0, 1.0], size=(n, n)) / np.sqrt(n)
        out[i] = svdvals(A)[-1]
    return out


def plot_convergence_with_n(save=True):
    # how fast does the (1+o(1)) factor go to 1?
    # plots P(sigma_n <= eps/sqrt(n)) / eps for various n
    dims = [20, 50, 100, 200, 500]
    eps_vals = [0.1, 0.3, 0.5, 0.8]
    n_trials = 2000

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, eps in enumerate(eps_vals):
        ax = axes[idx]
        ratios_gauss = []
        ratios_radem = []
        for n in dims:
            print(f"  eps={eps}, n={n}...")
            sv_g = compute_least_svs(n, n_trials, RNG, "gaussian")
            sv_r = compute_least_svs(n, n_trials, RNG, "rademacher")
            p_g = np.mean(sv_g <= eps / np.sqrt(n))
            p_r = np.mean(sv_r <= eps / np.sqrt(n))
            ratios_gauss.append(p_g / eps if eps > 0 else 0)
            ratios_radem.append(p_r / eps if eps > 0 else 0)

        ax.plot(dims, ratios_gauss, "bo-", label="Gaussian", markersize=5)
        ax.plot(dims, ratios_radem, "rs--", label="Rademacher", markersize=5)
        ax.axhline(y=1.0, color="k", linestyle=":", alpha=0.5, label="Ratio = 1")

        # o(1) envelope from SSS
        theory_upper = [1 + np.log(n)**(-1.0/16) for n in dims]
        ax.plot(dims, theory_upper, "g:", lw=1.5,
                label=r"$1 + (\log n)^{-1/16}$")

        ax.set_xlabel("n")
        ax.set_ylabel(r"$\mathbb{P}[\sigma_n \leq \varepsilon n^{-1/2}] / \varepsilon$")
        ax.set_title(rf"$\varepsilon = {eps}$")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    fig.suptitle(
        r"Convergence of $\mathbb{P}/\varepsilon$ Ratio to 1 (SSS Theorem 1.1)",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save:
        path = os.path.join(FIGDIR, "convergence_ratio_vs_n.png")
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    plt.close(fig)


def plot_error_term_decay(save=True):
    # |P/eps - 1| vs n, should decay like (log n)^{-1/16} or faster
    dims = [20, 50, 100, 200, 500]
    eps = 0.3
    n_trials = 2000

    errors_gauss = []
    errors_radem = []
    for n in dims:
        print(f"  error term: n={n}...")
        sv_g = compute_least_svs(n, n_trials, RNG, "gaussian")
        sv_r = compute_least_svs(n, n_trials, RNG, "rademacher")
        p_g = np.mean(sv_g <= eps / np.sqrt(n))
        p_r = np.mean(sv_r <= eps / np.sqrt(n))
        errors_gauss.append(abs(p_g / eps - 1))
        errors_radem.append(abs(p_r / eps - 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, errors_gauss, "bo-", label="Gaussian", markersize=6)
    ax.plot(dims, errors_radem, "rs--", label="Rademacher", markersize=6)

    # theory ref
    theory = [np.log(n)**(-1.0/16) for n in dims]
    ax.plot(dims, theory, "g:", lw=2, label=r"$(\log n)^{-1/16}$")

    ax.set_xlabel("n")
    ax.set_ylabel(r"$|\mathbb{P}/\varepsilon - 1|$")
    ax.set_title(rf"Decay of Error Term ($\varepsilon = {eps}$)")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(FIGDIR, "error_term_decay.png")
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    plt.close(fig)


def main():
    print("=== Perturbation / convergence analysis ===")
    plot_convergence_with_n()
    plot_error_term_decay()
    print("Done.\n")


if __name__ == "__main__":
    main()
