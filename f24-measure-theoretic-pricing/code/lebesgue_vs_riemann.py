# lebesgue vs riemann integration demos
# evans ch.1, folland ch.2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use("Agg")
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def approximate_dirichlet(x, max_denom=50):
    # brute force approx of 1_Q -- check closeness to p/q
    result = np.zeros_like(x)
    eps = 1e-10
    for q in range(1, max_denom + 1):
        for p in range(0, q + 1):
            result[np.abs(x - p / q) < eps] = 1.0
    return result


def fat_cantor_indicator(x, depth=8):
    # SVC set, measure = 1/2
    in_set = np.ones(len(x), dtype=bool)
    intervals = [(0.0, 1.0)]
    for k in range(1, depth + 1):
        removal_half_len = 1.0 / (2 * 4**k)
        new_intervals = []
        for (a, b) in intervals:
            mid = (a + b) / 2.0
            mask = (x > mid - removal_half_len) & (x < mid + removal_half_len)
            in_set[mask] = False
            new_intervals.append((a, mid - removal_half_len))
            new_intervals.append((mid + removal_half_len, b))
        intervals = new_intervals
    return in_set.astype(float)


def riemann_sum(f_vals, dx):
    return np.sum(f_vals[:-1]) * dx


def lebesgue_integral_approx(f_vals, dx, n_levels=200):
    # layer cake formula -- basically coarea in 1d
    f_min, f_max = np.min(f_vals), np.max(f_vals)
    if f_max - f_min < 1e-15:
        return f_min
    levels = np.linspace(f_min, f_max, n_levels + 1)
    dt = levels[1] - levels[0]
    integral = 0.0
    for t in levels[:-1]:
        mu_level = np.sum(f_vals > t) * dx
        integral += mu_level * dt
    return integral


def plot_dirichlet_comparison():
    ns = [100, 500, 1000, 5000, 10000, 50000]
    riemann_vals = []
    lebesgue_vals = []

    for n in ns:
        x = np.linspace(0, 1, n, endpoint=False)
        dx = 1.0 / n
        f = approximate_dirichlet(x, max_denom=50)
        riemann_vals.append(riemann_sum(f, dx))
        lebesgue_vals.append(lebesgue_integral_approx(f, dx))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    x_plot = np.linspace(0, 1, 2000, endpoint=False)
    f_plot = approximate_dirichlet(x_plot, max_denom=20)
    ax1.scatter(x_plot[f_plot > 0.5], f_plot[f_plot > 0.5],
                s=1, color="red", label="f=1 (rational)")
    ax1.scatter(x_plot[f_plot < 0.5], f_plot[f_plot < 0.5],
                s=0.3, color="blue", alpha=0.3, label="f=0 (irrational)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.set_title("Approximate Dirichlet function\n(rationals with denominator $\\leq 20$)")
    ax1.legend(markerscale=5)
    ax1.set_ylim(-0.1, 1.3)

    ax2.plot(ns, riemann_vals, "rs-", label="Riemann sum", markersize=5)
    ax2.plot(ns, lebesgue_vals, "bo-", label="Lebesgue (layer-cake)", markersize=5)
    ax2.axhline(0, color="green", linestyle="--", alpha=0.7, label="True integral = 0")
    ax2.set_xlabel("Number of grid points $n$")
    ax2.set_ylabel("Integral estimate")
    ax2.set_title("Integration of approximate Dirichlet function")
    ax2.legend()
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGDIR, "dirichlet_integration.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_fat_cantor():
    N = 50000
    x = np.linspace(0, 1, N, endpoint=False)
    dx = 1.0 / N
    f = fat_cantor_indicator(x, depth=8)

    resolutions = [500, 1000, 2000, 5000, 10000, 50000]
    riemann_upper = []
    riemann_lower = []
    lebesgue_vals = []

    for n in resolutions:
        xn = np.linspace(0, 1, n, endpoint=False)
        dxn = 1.0 / n
        fn = fat_cantor_indicator(xn, depth=8)

        n_blocks = min(n // 2, 500)
        block_size = n // n_blocks
        upper = 0.0
        lower = 0.0
        for i in range(n_blocks):
            block = fn[i * block_size:(i + 1) * block_size]
            upper += np.max(block) * (block_size * dxn)
            lower += np.min(block) * (block_size * dxn)
        riemann_upper.append(upper)
        riemann_lower.append(lower)
        lebesgue_vals.append(lebesgue_integral_approx(fn, dxn))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.fill_between(x, 0, f, alpha=0.5, color="purple", linewidth=0)
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$\mathbf{1}_{C}(x)$")
    ax1.set_title("Fat Cantor set indicator (SVC set, measure $\\approx 1/2$)")
    ax1.set_ylim(-0.05, 1.15)

    ax2.plot(resolutions, riemann_upper, "r^-", label="Riemann upper sum", markersize=5)
    ax2.plot(resolutions, riemann_lower, "rv-", label="Riemann lower sum", markersize=5)
    ax2.plot(resolutions, lebesgue_vals, "bo-", label="Lebesgue (layer-cake)", markersize=5)
    ax2.axhline(0.5, color="green", linestyle="--", alpha=0.7,
                label="True Lebesgue integral = 1/2")
    ax2.set_xlabel("Number of grid points $n$")
    ax2.set_ylabel("Integral estimate")
    ax2.set_title("Integration of fat Cantor set indicator")
    ax2.legend(fontsize=8)
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGDIR, "fat_cantor_integration.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_convergence_rates():
    ns = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000])

    true_sin = 2.0 / np.pi
    err_riemann_sin = []
    err_lebesgue_sin = []

    c = 1.0 / np.sqrt(2)
    true_step = 1.0 - c
    err_riemann_step = []
    err_lebesgue_step = []

    for n in ns:
        x = np.linspace(0, 1, n, endpoint=False)
        dx = 1.0 / n

        f_sin = np.sin(np.pi * x)
        err_riemann_sin.append(abs(riemann_sum(f_sin, dx) - true_sin))
        err_lebesgue_sin.append(abs(lebesgue_integral_approx(f_sin, dx) - true_sin))

        f_step = (x >= c).astype(float)
        err_riemann_step.append(abs(riemann_sum(f_step, dx) - true_step))
        err_lebesgue_step.append(abs(lebesgue_integral_approx(f_step, dx) - true_step))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.loglog(ns, err_riemann_sin, "rs-", label="Riemann", markersize=4)
    ax1.loglog(ns, err_lebesgue_sin, "bo-", label="Lebesgue", markersize=4)
    ax1.set_xlabel("Grid points $n$")
    ax1.set_ylabel("Absolute error")
    ax1.set_title(r"Smooth: $\sin(\pi x)$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.loglog(ns, err_riemann_step, "rs-", label="Riemann", markersize=4)
    ax2.loglog(ns, err_lebesgue_step, "bo-", label="Lebesgue", markersize=4)
    ax2.set_xlabel("Grid points $n$")
    ax2.set_ylabel("Absolute error")
    ax2.set_title(r"Discontinuous: step at $1/\sqrt{2}$")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGDIR, "convergence_rates_integration.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def run_all():
    print("=== Lebesgue vs Riemann ===")
    plot_dirichlet_comparison()
    plot_fat_cantor()
    plot_convergence_rates()


if __name__ == "__main__":
    run_all()
