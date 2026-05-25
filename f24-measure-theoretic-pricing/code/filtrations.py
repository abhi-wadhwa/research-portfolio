# sigma algebras + filtrations on binomial trees / brownian motion
# ch.1 motivation stuff

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use("Agg")  # need this or matplotlib crashes on mac
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def binomial_tree_paths(S0, u, d, T, n_paths):
    rng = np.random.default_rng(42)
    moves = rng.choice([u, d], size=(n_paths, T))
    paths = np.zeros((n_paths, T + 1))
    paths[:, 0] = S0
    for t in range(T):
        paths[:, t + 1] = paths[:, t] * moves[:, t]
    return paths


def plot_binomial_filtration():
    S0, u, d, T = 100.0, 1.1, 0.9, 4
    n_paths = 500  # 500 seems to look ok
    paths = binomial_tree_paths(S0, u, d, T, n_paths)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    fig.suptitle(
        r"Filtration $\{\mathcal{F}_t\}_{t=0}^{T}$: partitions refining over time",
        fontsize=13,
    )

    for idx, t in enumerate(range(1, T + 1)):
        ax = axes[idx]
        vals_t = np.round(paths[:, t], 4)
        unique_vals = np.sort(np.unique(vals_t))

        for v in unique_vals:
            mask = vals_t == v
            subset = paths[mask, : t + 1]
            for p in subset[:min(15, len(subset))]:
                ax.plot(range(t + 1), p, alpha=0.4, linewidth=0.7)

        ax.set_title(rf"$\mathcal{{F}}_{{{t}}}$  ({len(unique_vals)} atoms)")
        ax.set_xlabel("Time step")
        if idx == 0:
            ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGDIR, "filtration_binomial.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def brownian_conditional_expectations():
    # E[max(B_1,0) | F_t] at a few times
    rng = np.random.default_rng(123)
    N = 1000
    n_paths = 5000  # TODO try bumping this up
    dt = 1.0 / N
    dW = rng.normal(0, np.sqrt(dt), (n_paths, N))
    W = np.zeros((n_paths, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)

    times = np.linspace(0, 1, N + 1)
    payoff = np.maximum(W[:, -1], 0)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(30):
        ax.plot(times, W[i, :], alpha=0.08, color="steelblue", linewidth=0.5)

    # analytic formula not MC (way cleaner, see evans p.23)
    t_indices = [0, N // 8, N // 4, N // 2, 3 * N // 4, N]
    cond_exp_vals = []
    for ti in t_indices:
        t_val = times[ti]
        remaining = 1.0 - t_val
        if remaining < 1e-12:
            ce = payoff
        else:
            from scipy.stats import norm
            x = W[:, ti]
            sig = np.sqrt(remaining)
            # E[max(x + sig*Z, 0)] = x*Phi(x/sig) + sig*phi(x/sig)
            ce = x * norm.cdf(x / sig) + sig * norm.pdf(x / sig)
        cond_exp_vals.append((t_val, ce))

    for t_val, ce in cond_exp_vals:
        label = rf"$E[g(B_1)\mid\mathcal{{F}}_{{{t_val:.2f}}}]$"
        ax.axvline(t_val, color="gray", linestyle="--", alpha=0.3)

    # var should go up then hit terminal
    variances = [np.var(ce) for _, ce in cond_exp_vals]
    ax2 = ax.twinx()
    t_vals = [tv for tv, _ in cond_exp_vals]
    ax2.plot(t_vals, variances, "ro-", label=r"Var$(E[g|\mathcal{F}_t])$", zorder=5)
    ax2.set_ylabel(r"Var of $E[g(B_1)|\mathcal{F}_t]$", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Brownian motion sample paths")
    ax.set_title("Information refinement: conditional expectation under the natural filtration")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    path = os.path.join(FIGDIR, "filtration_brownian_cond_exp.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_sigma_algebra_partition():
    # dumb partition viz for 3-step binomial but it works
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.suptitle(r"Partition of $\Omega = \{HHH, HHT, HTH, HTT, THH, THT, TTH, TTT\}$", fontsize=12)

    outcomes = ["HHH", "HHT", "HTH", "HTT", "THH", "THT", "TTH", "TTT"]

    partitions = [
        [outcomes],
        [["HHH","HHT","HTH","HTT"], ["THH","THT","TTH","TTT"]],
        [["HHH","HHT"], ["HTH","HTT"], ["THH","THT"], ["TTH","TTT"]],
        [[o] for o in outcomes],
    ]

    colors = plt.cm.Set3(np.linspace(0, 1, 8))

    for idx, (ax, partition) in enumerate(zip(axes, partitions)):
        y_pos = 0
        for block in partition:
            for j, omega in enumerate(block):
                oi = outcomes.index(omega)
                ax.barh(y_pos, 1, color=colors[oi], edgecolor="black", linewidth=0.5)
                ax.text(0.5, y_pos, omega, ha="center", va="center", fontsize=7)
                y_pos += 1
            if block != partition[-1]:
                ax.axhline(y_pos - 0.5, color="black", linewidth=2)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(outcomes) - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$\mathcal{{F}}_{{{idx}}}$" + f"\n({len(partition)} atoms)")

    plt.tight_layout()
    path = os.path.join(FIGDIR, "sigma_algebra_partition.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def run_all():
    print("=== Filtrations ===")
    plot_binomial_filtration()
    brownian_conditional_expectations()
    plot_sigma_algebra_partition()


if __name__ == "__main__":
    run_all()
