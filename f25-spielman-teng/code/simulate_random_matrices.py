import numpy as np
from scipy import stats
from scipy.linalg import svdvals
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# dims to test -- bigger ones take a while
DIMS = [10, 20, 50, 100, 200, 500]
N_TRIALS = 2000
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)
RNG = np.random.default_rng(42)


def least_sv_gaussian(n, trials, rng):
    """least singular vals for iid gaussian matrices"""
    out = np.empty(trials)
    for i in range(trials):
        A = rng.standard_normal((n, n)) / np.sqrt(n)
        out[i] = svdvals(A)[-1]
    return out


def least_sv_rademacher(n, trials, rng):
    """same but +/-1 entries"""
    out = np.empty(trials)
    for i in range(trials):
        A = rng.choice([-1.0, 1.0], size=(n, n)) / np.sqrt(n)
        out[i] = svdvals(A)[-1]
    return out


def edelman_density(x):
    # copied formula from edelman 1988 / paper eq (3)
    return (x + 1.0) * np.exp(-0.5 * x**2 - x)


def plot_histograms(save=True):
    data = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for idx, n in enumerate(DIMS):
        print(f"  n={n}...")
        sv_gauss = least_sv_gaussian(n, N_TRIALS, RNG)
        sv_radem = least_sv_rademacher(n, N_TRIALS, RNG)
        data[n] = {"gaussian": sv_gauss, "rademacher": sv_radem}

        ax = axes[idx]
        ax.hist(sv_gauss, bins=50, density=True, alpha=0.6, label="Gaussian")
        ax.hist(sv_radem, bins=50, density=True, alpha=0.6, label="Rademacher")
        ax.set_title(f"n = {n}")
        ax.set_xlabel(r"$\sigma_{\min}(A/\sqrt{n})$")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Least Singular Value: Gaussian vs Rademacher",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        path = os.path.join(FIGDIR, "least_sv_histograms.png")
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    plt.close(fig)
    return data


def plot_edelman_comparison(data, save=True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    big_dims = [50, 200, 500]

    for idx, n in enumerate(big_dims):
        ax = axes[idx]
        sv = data[n]["gaussian"]
        scaled = n * sv  # n * sigma_min should converge to edelman dist
        ax.hist(scaled, bins=60, density=True, alpha=0.5, label="Empirical")
        xgrid = np.linspace(0, max(scaled.max(), 6), 300)
        ax.plot(xgrid, edelman_density(xgrid), "r-", lw=2, label="Edelman density")
        ax.set_title(f"n = {n}")
        ax.set_xlabel(r"$n \cdot \sigma_{\min}$")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 8)

    fig.suptitle(
        r"Scaled Least Singular Value vs Edelman Density",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save:
        path = os.path.join(FIGDIR, "edelman_comparison.png")
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    plt.close(fig)


def plot_sss_bound_verification(data, save=True):
    # check P(sigma_n <= eps/sqrt(n)) <= (1+o(1))*eps -- see SSS thm 1.1
    epsilons = np.linspace(0.01, 1.0, 50)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    test_dims = [50, 200, 500]

    for idx, n in enumerate(test_dims):
        ax = axes[idx]
        sv = data[n]["gaussian"]
        emp_probs = np.array([np.mean(sv <= eps / np.sqrt(n)) for eps in epsilons])
        ax.plot(epsilons, emp_probs, "b-", lw=2, label="Empirical CDF")
        ax.plot(epsilons, epsilons, "r--", lw=1.5, label=r"$\varepsilon$ (ideal)")
        # (1+o(1)) envelope -- delta_n approx (log n)^{-1/16}
        delta_n = np.log(n)**(-1.0/16)
        ax.plot(epsilons, (1 + delta_n) * epsilons, 'g:', lw=1.5,
                label=rf"$(1+\delta_n)\varepsilon$, $\delta_n$={delta_n:.3f}")
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(r"$\mathbb{P}[\sigma_n \leq \varepsilon / \sqrt{n}]$")
        ax.set_title(f"n = {n}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1)

    fig.suptitle(
        r"Verification of SSS Bound: $\mathbb{P}[\sigma_n \leq \varepsilon n^{-1/2}] \leq (1+o(1))\varepsilon$",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save:
        path = os.path.join(FIGDIR, "sss_bound_verification.png")
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    plt.close(fig)


def plot_universality(data, save=True):
    # gauss vs rademacher CDFs -- thm 1.2
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    test_dims = [50, 200, 500]
    epsilons = np.linspace(0.01, 1.5, 80)

    for idx, n in enumerate(test_dims):
        ax = axes[idx]
        sv_g = data[n]["gaussian"]
        sv_r = data[n]["rademacher"]
        cdf_g = np.array([np.mean(sv_g <= eps / np.sqrt(n)) for eps in epsilons])
        cdf_r = np.array([np.mean(sv_r <= eps / np.sqrt(n)) for eps in epsilons])
        ax.plot(epsilons, cdf_g, "b-", lw=2, label="Gaussian")
        ax.plot(epsilons, cdf_r, "r--", lw=2, label="Rademacher")
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(r"$\mathbb{P}[\sigma_n \leq \varepsilon / \sqrt{n}]$")
        ax.set_title(f"n = {n}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Universality: Gaussian vs Rademacher Least Singular Value CDF", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save:
        path = os.path.join(FIGDIR, "universality_comparison.png")
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    plt.close(fig)


def plot_prob_bounds(data, save=True):
    # tail prob vs dim for several eps
    epsilons = [0.001, 0.005, 0.01, 0.05, 0.1]
    fig, ax = plt.subplots(figsize=(8, 5))

    for eps in epsilons:
        probs = []
        for n in DIMS:
            sv = data[n]["gaussian"]
            probs.append(np.mean(sv < eps / np.sqrt(n)))
        ax.plot(DIMS, probs, "o-", label=rf"$\varepsilon = {eps}$")

    # RV bound ref: C*eps + exp(-cn)
    for eps in [0.01, 0.1]:
        bound = [min(1.0, 2.0 * eps + np.exp(-0.1 * n)) for n in DIMS]
        ax.plot(DIMS, bound, "--", color="gray", alpha=0.5)

    ax.set_xlabel("Dimension n")
    ax.set_ylabel(r"$\mathbb{P}[\sigma_n < \varepsilon n^{-1/2}]$")
    ax.set_title("Tail Probability vs Dimension (Gaussian)")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(FIGDIR, "prob_bounds_vs_n.png")
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    plt.close(fig)


def main():
    print("=== Simulating least singular values ===")
    data = plot_histograms()
    plot_edelman_comparison(data)
    plot_sss_bound_verification(data)
    plot_universality(data)
    plot_prob_bounds(data)
    print("Done.\n")


if __name__ == "__main__":
    main()
