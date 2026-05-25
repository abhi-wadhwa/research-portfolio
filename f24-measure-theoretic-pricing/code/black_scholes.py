# black-scholes from scratch
# radon-nikodym / girsanov -- ties back to evans ch.1

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib
import os

matplotlib.use("Agg")
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def bs_call(S, K, T, r, sigma):
    if T < 1e-12:
        return np.maximum(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    # put-call parity but whatever
    if T < 1e-12:
        return np.maximum(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma):
    if T < 1e-12:
        return (S > K).astype(float)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


# vega, might need later
# def bs_vega(S, K, T, r, sigma):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#     return S * norm.pdf(d1) * np.sqrt(T)


def simulate_gbm(S0, mu, sigma, r, T, N, n_paths, rng):
    dt = T / N
    t = np.linspace(0, T, N + 1)

    dW_P = rng.normal(0, np.sqrt(dt), (n_paths, N))
    W_P = np.zeros((n_paths, N + 1))
    W_P[:, 1:] = np.cumsum(dW_P, axis=1)

    theta = (mu - r) / sigma  # market price of risk

    dW_Q = dW_P + theta * dt

    # log price under P (avoids numerical blowup)
    log_S_P = np.zeros((n_paths, N + 1))
    log_S_P[:, 0] = np.log(S0)
    for i in range(N):
        log_S_P[:, i + 1] = (log_S_P[:, i]
                              + (mu - 0.5 * sigma**2) * dt
                              + sigma * dW_P[:, i])
    S_P = np.exp(log_S_P)

    # Q paths -- just swap in the Q drift
    log_S_Q = np.zeros((n_paths, N + 1))
    log_S_Q[:, 0] = np.log(S0)
    for i in range(N):
        log_S_Q[:, i + 1] = (log_S_Q[:, i]
                               + (r - 0.5 * sigma**2) * dt
                               + sigma * dW_Q[:, i])
    S_Q = np.exp(log_S_Q)

    # dQ/dP exponential martingale
    radon_nikodym = np.exp(-theta * W_P[:, -1] - 0.5 * theta**2 * T)

    return t, S_P, S_Q, W_P, radon_nikodym


def plot_paths_P_vs_Q():
    S0, mu, sigma, r, T, N = 100.0, 0.12, 0.25, 0.05, 1.0, 252
    n_paths = 200
    rng = np.random.default_rng(42)

    t, S_P, S_Q, W_P, Z_T = simulate_gbm(S0, mu, sigma, r, T, N, n_paths, rng)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for i in range(min(80, n_paths)):
        ax1.plot(t, S_P[i, :], alpha=0.15, color="steelblue", linewidth=0.5)
        ax2.plot(t, S_Q[i, :], alpha=0.15, color="darkorange", linewidth=0.5)

    ax1.plot(t, np.mean(S_P, axis=0), "k-", linewidth=2, label=rf"Mean ($\mu={mu}$)")
    ax1.plot(t, S0 * np.exp(mu * t), "r--", linewidth=1.5, label=r"$S_0 e^{\mu t}$")
    ax1.set_title(r"Physical measure $\mathbb{P}$: $dS = \mu S\,dt + \sigma S\,dW^{\mathbb{P}}$")
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Stock price $S_t$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, np.mean(S_Q, axis=0), "k-", linewidth=2, label=rf"Mean ($r={r}$)")
    ax2.plot(t, S0 * np.exp(r * t), "r--", linewidth=1.5, label=r"$S_0 e^{rt}$")
    ax2.set_title(r"Risk-neutral measure $\mathbb{Q}$: $dS = r S\,dt + \sigma S\,dW^{\mathbb{Q}}$")
    ax2.set_xlabel("Time (years)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGDIR, "paths_P_vs_Q.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_girsanov_radon_nikodym():
    S0, mu, sigma, r, T, N = 100.0, 0.12, 0.25, 0.05, 1.0, 252
    n_paths = 50000
    rng = np.random.default_rng(123)

    t, S_P, S_Q, W_P, Z_T = simulate_gbm(S0, mu, sigma, r, T, N, n_paths, rng)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.hist(Z_T, bins=100, density=True, alpha=0.7, color="mediumpurple",
             edgecolor="black", linewidth=0.3)
    ax1.axvline(np.mean(Z_T), color="red", linestyle="--",
                label=rf"$E_{{\mathbb{{P}}}}[Z_T] = {np.mean(Z_T):.4f}$")
    ax1.set_xlabel(r"$Z_T = d\mathbb{Q}/d\mathbb{P}$")
    ax1.set_ylabel("Density")
    ax1.set_title("Radon-Nikodym derivative $Z_T$\n(Girsanov exponential martingale)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # sanity check: MC via P w/ importance weights = MC via Q = BS closed form
    K = 100.0
    payoff_P = np.maximum(S_P[:, -1] - K, 0)
    payoff_Q = np.maximum(S_Q[:, -1] - K, 0)

    mc_price_via_P = np.exp(-r * T) * np.mean(Z_T * payoff_P)
    mc_price_via_Q = np.exp(-r * T) * np.mean(payoff_Q)
    bsPrice = bs_call(S0, K, T, r, sigma)  # idk why i named this camelCase

    ax2.bar(["MC via P\n" + r"$e^{-rT}E_{\mathbb{P}}[Z_T \cdot g(S_T)]$",
             "MC via Q\n" + r"$e^{-rT}E_{\mathbb{Q}}[g(S_T)]$",
             "Black-Scholes\nclosed-form"],
            [mc_price_via_P, mc_price_via_Q, bsPrice],
            color=["steelblue", "darkorange", "forestgreen"],
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("European call price")
    ax2.set_title(f"Option pricing verification (K={K}, T={T})")
    for i, v in enumerate([mc_price_via_P, mc_price_via_Q, bsPrice]):
        ax2.text(i, v + 0.3, f"${v:.4f}", ha="center", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(FIGDIR, "girsanov_radon_nikodym.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_replicating_strategy():
    # delta hedging, pretty sure this is right
    S0, K, sigma, r, T, N = 100.0, 100.0, 0.25, 0.05, 1.0, 252
    rng = np.random.default_rng(2024)
    dt = T / N
    t = np.linspace(0, T, N + 1)

    dW = rng.normal(0, np.sqrt(dt), N)
    S = np.zeros(N + 1)
    S[0] = S0
    for i in range(N):
        S[i + 1] = S[i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW[i])

    tau = T - t
    delta = np.zeros(N + 1)
    for i in range(N + 1):
        if tau[i] < 1e-12:
            delta[i] = 1.0 if S[i] > K else 0.0
        else:
            delta[i] = bs_delta(S[i], K, tau[i], r, sigma)

    # replicate by delta hedging -- error shrinks w/ dt
    V = np.zeros(N + 1)
    V[0] = bs_call(S0, K, T, r, sigma)
    for i in range(N):
        cash = V[i] - delta[i] * S[i]
        V[i + 1] = delta[i] * S[i + 1] + cash * np.exp(r * dt)

    bs_vals = np.array([bs_call(S[i], K, tau[i], r, sigma) for i in range(N + 1)])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax1.plot(t, S, "b-", linewidth=1)
    ax1.axhline(K, color="gray", linestyle="--", alpha=0.5, label=f"Strike K={K}")
    ax1.set_ylabel("Stock price $S_t$")
    ax1.set_title("Martingale representation: delta-hedging replication")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, bs_vals, "g-", linewidth=1.5, label="BS option value $C(S_t, t)$")
    ax2.plot(t, V, "r--", linewidth=1, label="Replicating portfolio $V_t$")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(t, delta, "purple", linewidth=1)
    ax3.set_xlabel("Time (years)")
    ax3.set_ylabel(r"$\Delta_t = \partial C/\partial S$")
    ax3.set_title("Hedge ratio (delta)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGDIR, "replicating_strategy.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_option_surface():
    # 3d surface, kinda slow but looks cool
    S_range = np.linspace(50, 150, 100)
    T_range = np.linspace(0.01, 2.0, 100)
    S_grid, T_grid = np.meshgrid(S_range, T_range)

    K, r, sigma = 100.0, 0.05, 0.25
    C = np.zeros_like(S_grid)
    for i in range(S_grid.shape[0]):
        for j in range(S_grid.shape[1]):
            C[i, j] = bs_call(S_grid[i, j], K, T_grid[i, j], r, sigma)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(S_grid, T_grid, C, cmap="viridis", alpha=0.85,
                           edgecolor="none")
    ax.set_xlabel("Spot price $S$")
    ax.set_ylabel("Time to maturity $T$")
    ax.set_zlabel("Call price $C(S, T)$")
    ax.set_title(f"European call price surface\n($K={K}$, $r={r}$, $\\sigma={sigma}$)")
    fig.colorbar(surf, shrink=0.5, aspect=10, label="Price")
    ax.view_init(elev=25, azim=135)

    plt.tight_layout()
    path = os.path.join(FIGDIR, "option_price_surface.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_convergence_mc_pricing():
    # should see 1/sqrt(n) rate
    S0, K, sigma, r, T = 100.0, 100.0, 0.25, 0.05, 1.0
    bs_true = bs_call(S0, K, T, r, sigma)

    rng = np.random.default_rng(777)
    max_paths = 100000
    Z = rng.standard_normal(max_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.exp(-r * T) * np.maximum(ST - K, 0)

    ns = np.logspace(1, 5, 200).astype(int)
    ns = np.unique(ns)

    means = np.array([np.mean(payoffs[:n]) for n in ns])
    stds = np.array([np.std(payoffs[:n]) / np.sqrt(n) for n in ns])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.fill_between(ns, means - 1.96 * stds, means + 1.96 * stds,
                     alpha=0.3, color="steelblue", label="95% CI")
    ax1.plot(ns, means, "b-", linewidth=0.8, label="MC estimate")
    ax1.axhline(bs_true, color="red", linestyle="--",
                label=f"BS = \${bs_true:.4f}")
    ax1.set_xlabel("Number of paths")
    ax1.set_ylabel("Price estimate")
    ax1.set_title("Monte Carlo convergence for European call")
    ax1.set_xscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.loglog(ns, np.abs(means - bs_true), "b-", alpha=0.5, linewidth=0.8,
               label="|MC - BS|")
    ax2.loglog(ns, stds, "r-", linewidth=1.2, label=r"Std error $\sigma/\sqrt{n}$")
    ax2.loglog(ns, 10.0 / np.sqrt(ns), "k--", alpha=0.5, label=r"$O(1/\sqrt{n})$")
    ax2.set_xlabel("Number of paths")
    ax2.set_ylabel("Error")
    ax2.set_title("Convergence rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGDIR, "mc_convergence.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def run_all():
    print("=== Black-Scholes ===")
    plot_paths_P_vs_Q()
    plot_girsanov_radon_nikodym()
    plot_replicating_strategy()
    plot_option_surface()
    plot_convergence_mc_pricing()


if __name__ == "__main__":
    run_all()
