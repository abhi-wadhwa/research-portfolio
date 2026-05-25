# convergence modes -- Lp, doob, typewriter etc
# evans ch.1 + billingsley

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use("Agg")
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def lp_convergence_demo():
    rng = np.random.default_rng(314)
    n_samples = 50000
    X = rng.standard_normal(n_samples)

    ns = np.arange(1, 201)
    ps = [1, 2, 4, 8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    from scipy.special import gamma as gamma_fn
    def lp_norm_std_normal(p):
        return (2**(p / 2) * gamma_fn((p + 1) / 2) / np.sqrt(np.pi))**(1 / p)

    for p in ps:
        empirical_norms = []
        for n in ns:
            Z_n = rng.standard_normal(n_samples)
            diff = Z_n / np.sqrt(n)
            lp_emp = np.mean(np.abs(diff)**p)**(1 / p)
            empirical_norms.append(lp_emp)

        theoretical = [lp_norm_std_normal(p) / np.sqrt(n) for n in ns]
        ax1.plot(ns, empirical_norms, alpha=0.6, linewidth=0.8,
                 label=rf"$\|X_n - X\|_{{{p}}}$ (empirical)")
        ax1.plot(ns, theoretical, "--", alpha=0.8, linewidth=1.2,
                 label=rf"$C_{{{p}}}/\sqrt{{n}}$ (theory)")

    ax1.set_xlabel("$n$")
    ax1.set_ylabel(r"$\|X_n - X\|_p$")
    ax1.set_title(r"$L^p$ convergence: $X_n = X + Z_n/\sqrt{n}$")
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # L1 conv but NOT L2 -- X_n = n * 1_{U < 1/n^2}
    # E[|X_n|] = 1/n -> 0 but E[X_n^2] = 1 forever lol
    ns2 = np.arange(1, 501)
    l1_norms = []
    l2_norms = []
    for n in ns2:
        U = rng.uniform(0, 1, n_samples)
        Xn = n * (U < 1.0 / n**2).astype(float)
        l1_norms.append(np.mean(np.abs(Xn)))
        l2_norms.append(np.mean(Xn**2)**0.5)

    ax2.plot(ns2, l1_norms, "b-", alpha=0.7, linewidth=0.8, label=r"$\|X_n\|_1$")
    ax2.plot(ns2, l2_norms, "r-", alpha=0.7, linewidth=0.8, label=r"$\|X_n\|_2$")
    ax2.axhline(0, color="green", linestyle="--", alpha=0.5)
    ax2.set_xlabel("$n$")
    ax2.set_ylabel("Norm")
    ax2.set_title(r"$L^1$ convergence $\not\Rightarrow$ $L^2$ convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGDIR, "lp_convergence.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def doob_martingale_convergence():
    # M_t = E[|B_1| | F_t], should converge a.s. + L1
    rng = np.random.default_rng(2024)
    N = 500
    n_paths = 3000
    dt = 1.0 / N

    increments = rng.choice([-1, 1], size=(n_paths, N)) * np.sqrt(dt)
    walk = np.zeros((n_paths, N + 1))
    walk[:, 1:] = np.cumsum(increments, axis=1)

    X = np.abs(walk[:, -1])

    from scipy.stats import norm
    times = np.arange(N + 1) / N

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # analytic -- way better than binning paths
    # condition on B_1 = B_t + sqrt(1-t)*Z
    martingale = np.zeros((n_paths, N + 1))
    for t_idx in range(N + 1):
        t = times[t_idx]
        remaining = 1.0 - t
        x = walk[:, t_idx]
        if remaining < 1e-12:
            martingale[:, t_idx] = np.abs(x)
        else:
            sig = np.sqrt(remaining)
            martingale[:, t_idx] = (
                x * (2 * norm.cdf(x / sig) - 1) + 2 * sig * norm.pdf(x / sig)
            )

    for i in range(50):
        ax1.plot(times, martingale[i, :], alpha=0.15, color="steelblue", linewidth=0.5)

    ax1.plot(times, np.mean(martingale, axis=0), "k-", linewidth=2,
             label=r"$\bar{M}_n = $ sample mean")
    ax1.axhline(np.mean(X), color="red", linestyle="--",
                label=rf"$E[|B_1|] = {np.mean(X):.3f}$")
    ax1.set_xlabel("Time $t$")
    ax1.set_ylabel(r"$M_t = E[|B_1| \mid \mathcal{F}_t]$")
    ax1.set_title("Doob martingale convergence:\nsample paths of $E[|B_1|\\mid\\mathcal{F}_t]$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    l1_errors = []
    checkpoints = list(range(0, N + 1, 5))
    for t_idx in checkpoints:
        l1_errors.append(np.mean(np.abs(martingale[:, t_idx] - X)))

    ax2.plot([times[t] for t in checkpoints], l1_errors, "ro-", markersize=3)
    ax2.set_xlabel("Time $t$")
    ax2.set_ylabel(r"$E[|M_t - M_\infty|]$")
    ax2.set_title(r"$L^1$ convergence: $E[|M_t - |B_1||] \to 0$")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGDIR, "martingale_convergence.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def convergence_modes_comparison():
    # typewriter always confuses me ngl
    rng = np.random.default_rng(999)
    n_samples = 10000
    ns = np.arange(1, 301)

    Z = rng.standard_normal(n_samples)
    U = rng.uniform(0, 1, n_samples)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # a.s. convergence: X_n = Z/n
    ax = axes[0]
    paths_as = np.outer(Z[:20], 1.0 / ns)
    for i in range(20):
        ax.plot(ns, paths_as[i, :], alpha=0.4, linewidth=0.7)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("$n$")
    ax.set_ylabel("$X_n$")
    ax.set_title(r"A.s. convergence: $X_n = Z/n \to 0$")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    l2_norms_as = [np.mean(Z**2 / n**2)**0.5 for n in ns]
    ax.plot(ns, l2_norms_as, "b-", label=r"$\|Z/n\|_2$")
    ax.plot(ns, np.mean(np.abs(Z)) / ns, "r--", label=r"$\|Z/n\|_1$")
    ax.set_xlabel("$n$")
    ax.set_ylabel("Norm")
    ax.set_title(r"$L^p$ norms of $X_n = Z/n$")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # typewriter -- converges in prob but not a.s.
    # see billingsley exercise 20.5
    ax = axes[2]
    prob_exceed = []
    for n in range(1, 301):
        k = int(np.floor(np.log2(n + 1)))
        m = n - 2**k + 1
        length = 1.0 / 2**k
        a_n = m * length
        b_n = a_n + length
        p = np.mean((U >= a_n) & (U < b_n))
        prob_exceed.append(p)

    ax.plot(range(1, 301), prob_exceed, "g-", alpha=0.7, linewidth=0.8)
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$P(X_n > 0)$")
    ax.set_title("Typewriter sequence:\nconverges in prob. but not a.s.")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGDIR, "convergence_modes.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def run_all():
    print("=== Convergence theorems ===")
    lp_convergence_demo()
    doob_martingale_convergence()
    convergence_modes_comparison()


if __name__ == "__main__":
    run_all()
