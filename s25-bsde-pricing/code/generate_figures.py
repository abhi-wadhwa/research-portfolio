import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sde_simulation import (gbm_euler_maruyama, gbm_milstein, gbm_exact,
                             cev_euler_maruyama, heston_euler_maruyama,
                             strong_error, weak_error)
from bsde_solver import (solve_bsde, european_call_payoff,
                          linear_driver, transaction_cost_driver,
                          funding_cost_driver, uncertain_vol_driver)
from pde_solver import (solve_linear_bs_cn, solve_nonlinear_pde_cn,
                         transaction_cost_nl, funding_cost_nl,
                         uncertain_vol_nl)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)
plt.rcParams.update({
    'figure.figsize': (7, 5),
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.family': 'serif',
})

# params
S0 = 100.0
K = 100.0
r = 0.05
sigma = 0.2
T = 1.0
rng_seed = 42


def fig1_bsde_vs_pde():
    # feynman-kac check: bsde vs pde for linear BS
    print("Generating Figure 1: BSDE vs PDE comparison...")

    S_pde, t_pde, U_pde, price_pde = solve_linear_bs_cn(
        S0, K, r, sigma, T, N_S=200, N_t=500, option_type='call')

    M = 50000
    N = 100
    spots = np.linspace(60, 140, 25)
    bsde_prices = []

    for s in spots:
        rng = np.random.default_rng(rng_seed)
        t_fwd, X = gbm_euler_maruyama(s, r, sigma, T, N, M, rng=rng)
        Y, Z = solve_bsde(X, t_fwd, european_call_payoff(K),
                           linear_driver(r), basis_degree=4)
        bsde_prices.append(np.mean(Y[:, 0]))

    bsde_prices = np.array(bsde_prices)
    pde_prices = np.interp(spots, S_pde, U_pde[0, :])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(spots, pde_prices, 'b-', linewidth=2, label='PDE (Crank-Nicolson)')
    axes[0].plot(spots, bsde_prices, 'ro', markersize=5, label='BSDE (Monte Carlo)')
    axes[0].set_xlabel('Spot Price $S_0$')
    axes[0].set_ylabel('Option Price')
    axes[0].set_title('European Call: BSDE vs PDE')
    axes[0].legend()

    abs_err = np.abs(bsde_prices - pde_prices)
    axes[1].plot(spots, abs_err, 'k-o', markersize=4)
    axes[1].set_xlabel('Spot Price $S_0$')
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title('|BSDE - PDE| Error')
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_bsde_vs_pde.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig1_bsde_vs_pde.png'))
    plt.close()
    print("  Done.")


def fig2_price_surfaces():
    # price surfaces under different nonlinearities
    print("Generating Figure 2: Price surfaces...")

    N_S, N_t = 200, 500
    S_pde, t_pde, U_linear, _ = solve_linear_bs_cn(
        S0, K, r, sigma, T, N_S=N_S, N_t=N_t, option_type='call')

    _, _, U_tc, _ = solve_nonlinear_pde_cn(
        S0, K, r, sigma, T, transaction_cost_nl(r, 0.01),
        N_S=N_S, N_t=N_t, option_type='call', picard_iter=4)

    _, _, U_fc, _ = solve_nonlinear_pde_cn(
        S0, K, r, sigma, T, funding_cost_nl(0.03, 0.08, r),
        N_S=N_S, N_t=N_t, option_type='call', picard_iter=4)

    _, _, U_uv, _ = solve_nonlinear_pde_cn(
        S0, K, r, sigma, T, uncertain_vol_nl(0.15, 0.30, sigma),
        N_S=N_S, N_t=N_t, option_type='call', picard_iter=4)

    fig, ax = plt.subplots(figsize=(8, 5))
    mask = (S_pde >= 60) & (S_pde <= 140)
    ax.plot(S_pde[mask], U_linear[0, mask], 'b-', linewidth=2,
            label='Linear (Black-Scholes)')
    ax.plot(S_pde[mask], U_tc[0, mask], 'r--', linewidth=2,
            label='Transaction costs ($\\kappa=0.01$)')
    ax.plot(S_pde[mask], U_fc[0, mask], 'g-.', linewidth=2,
            label='Funding costs ($r_b=8\\%, r_l=3\\%$)')
    ax.plot(S_pde[mask], U_uv[0, mask], 'm:', linewidth=2,
            label='Uncertain vol ($\\sigma \\in [0.15, 0.30]$)')
    ax.set_xlabel('Spot Price $S$')
    ax.set_ylabel('Option Price $u(0, S)$')
    ax.set_title('European Call Price Under Market Frictions')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_price_surfaces.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig2_price_surfaces.png'))
    plt.close()
    print("  Done.")


def fig3_bsde_convergence():
    # bsde error vs num time steps
    print("Generating Figure 3: BSDE convergence...")

    from scipy.stats import norm
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    M = 80000
    N_values = [20, 40, 80, 160, 320]
    errors = []

    for N in N_values:
        rng = np.random.default_rng(rng_seed)
        t_fwd, X = gbm_euler_maruyama(S0, r, sigma, T, N, M, rng=rng)
        Y, Z = solve_bsde(X, t_fwd, european_call_payoff(K),
                           linear_driver(r), basis_degree=4)
        price = np.mean(Y[:, 0])
        errors.append(np.abs(price - bs_price))
        print(f"    N={N:4d}: BSDE price={price:.4f}, BS={bs_price:.4f}, "
              f"error={errors[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(N_values, errors, 'bo-', linewidth=2, markersize=8, label='BSDE error')
    ref = errors[0] * (np.array(N_values, dtype=float) / N_values[0])**(-0.5)
    ax.loglog(N_values, ref, 'k--', alpha=0.5, label='$O(N^{-1/2})$ reference')
    ax.set_xlabel('Number of Time Steps $N$')
    ax.set_ylabel('Absolute Error $|\\hat{u} - u_{BS}|$')
    ax.set_title('BSDE Solver Convergence')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_bsde_convergence.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig3_bsde_convergence.png'))
    plt.close()
    print("  Done.")


def fig4_sde_convergence():
    # strong + weak convergence rates for EM and milstein
    print("Generating Figure 4: SDE convergence rates...")

    M = 100000
    N_ref = 2048
    N_values = [16, 32, 64, 128, 256, 512]

    strong_em = []
    strong_mil = []
    weak_em = []
    weak_mil = []

    rng_ref = np.random.default_rng(rng_seed)
    _, S_ref = gbm_exact(S0, r, sigma, T, N_ref, M, rng=rng_ref)

    for N in N_values:
        step_ratio = N_ref // N
        S_ref_sub = S_ref[:, ::step_ratio]

        rng_em = np.random.default_rng(rng_seed)
        _, S_em = gbm_euler_maruyama(S0, r, sigma, T, N, M, rng=rng_em)
        rng_mil = np.random.default_rng(rng_seed)
        _, S_mil = gbm_milstein(S0, r, sigma, T, N, M, rng=rng_mil)

        strong_em.append(strong_error(S_em, S_ref_sub))
        strong_mil.append(strong_error(S_mil, S_ref_sub))
        weak_em.append(weak_error(S_em, S_ref_sub))
        weak_mil.append(weak_error(S_mil, S_ref_sub))

    dt_values = T / np.array(N_values, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # strong
    axes[0].loglog(dt_values, strong_em, 'bo-', label='Euler-Maruyama', linewidth=2)
    axes[0].loglog(dt_values, strong_mil, 'rs-', label='Milstein', linewidth=2)
    ref05 = strong_em[0] * (dt_values / dt_values[0])**0.5
    ref10 = strong_mil[0] * (dt_values / dt_values[0])**1.0
    axes[0].loglog(dt_values, ref05, 'b--', alpha=0.4, label='$O(\\Delta t^{0.5})$')
    axes[0].loglog(dt_values, ref10, 'r--', alpha=0.4, label='$O(\\Delta t^{1.0})$')
    axes[0].set_xlabel('$\\Delta t$')
    axes[0].set_ylabel('Strong Error')
    axes[0].set_title('Strong Convergence')
    axes[0].legend(fontsize=9)

    # weak
    axes[1].loglog(dt_values, weak_em, 'bo-', label='Euler-Maruyama', linewidth=2)
    axes[1].loglog(dt_values, weak_mil, 'rs-', label='Milstein', linewidth=2)
    ref10w = weak_em[0] * (dt_values / dt_values[0])**1.0
    axes[1].loglog(dt_values, ref10w, 'k--', alpha=0.4, label='$O(\\Delta t^{1.0})$')
    axes[1].set_xlabel('$\\Delta t$')
    axes[1].set_ylabel('Weak Error')
    axes[1].set_title('Weak Convergence')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_sde_convergence.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig4_sde_convergence.png'))
    plt.close()
    print("  Done.")


def fig5_transaction_costs():
    # tc impact on option price
    print("Generating Figure 5: Transaction cost impact...")

    N_S, N_t = 200, 500
    kappa_values = [0.0, 0.005, 0.01, 0.02, 0.05]
    S_pde, t_pde, _, _ = solve_linear_bs_cn(S0, K, r, sigma, T,
                                              N_S=N_S, N_t=N_t)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    mask = (S_pde >= 70) & (S_pde <= 130)
    colors = sns.color_palette('viridis', len(kappa_values))

    prices_at_S0 = []
    for i, kap in enumerate(kappa_values):
        if kap == 0:
            _, _, U, p = solve_linear_bs_cn(S0, K, r, sigma, T,
                                             N_S=N_S, N_t=N_t)
        else:
            _, _, U, p = solve_nonlinear_pde_cn(
                S0, K, r, sigma, T, transaction_cost_nl(r, kap),
                N_S=N_S, N_t=N_t, option_type='call', picard_iter=4)
        prices_at_S0.append(p)
        axes[0].plot(S_pde[mask], U[0, mask], color=colors[i], linewidth=2,
                     label=f'$\\kappa = {kap}$')

    axes[0].set_xlabel('Spot Price $S$')
    axes[0].set_ylabel('Option Price')
    axes[0].set_title('Call Price vs Transaction Cost Level')
    axes[0].legend(fontsize=9)

    axes[1].plot(kappa_values, prices_at_S0, 'ko-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Transaction Cost Parameter $\\kappa$')
    axes[1].set_ylabel('ATM Call Price at $S_0=100$')
    axes[1].set_title('ATM Price Sensitivity to $\\kappa$')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig5_transaction_costs.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig5_transaction_costs.png'))
    plt.close()
    print("  Done.")


def fig6_uncertain_vol():
    # uncertain vol bounds
    print("Generating Figure 6: Uncertain volatility bounds...")

    N_S, N_t = 200, 500

    _, _, U_low, _ = solve_linear_bs_cn(S0, K, r, 0.15, T, N_S=N_S, N_t=N_t)
    S_pde, _, U_mid, _ = solve_linear_bs_cn(S0, K, r, sigma, T, N_S=N_S, N_t=N_t)
    _, _, U_high, _ = solve_linear_bs_cn(S0, K, r, 0.30, T, N_S=N_S, N_t=N_t)

    _, _, U_uv, _ = solve_nonlinear_pde_cn(
        S0, K, r, sigma, T, uncertain_vol_nl(0.15, 0.30, sigma),
        N_S=N_S, N_t=N_t, option_type='call', picard_iter=4)

    fig, ax = plt.subplots(figsize=(8, 5))
    mask = (S_pde >= 70) & (S_pde <= 130)

    ax.fill_between(S_pde[mask], U_low[0, mask], U_high[0, mask],
                     alpha=0.2, color='gray', label='BS range $[\\sigma_L, \\sigma_H]$')
    ax.plot(S_pde[mask], U_mid[0, mask], 'b-', linewidth=2,
            label='BS ($\\sigma=0.20$)')
    ax.plot(S_pde[mask], U_uv[0, mask], 'r--', linewidth=2,
            label='Uncertain vol (worst-case)')
    ax.plot(S_pde[mask], U_high[0, mask], 'k:', linewidth=1,
            label='BS ($\\sigma=0.30$)')
    ax.plot(S_pde[mask], U_low[0, mask], 'k-.', linewidth=1,
            label='BS ($\\sigma=0.15$)')
    ax.set_xlabel('Spot Price $S$')
    ax.set_ylabel('Option Price')
    ax.set_title('Uncertain Volatility: Worst-Case Pricing Bounds')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig6_uncertain_vol.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig6_uncertain_vol.png'))
    plt.close()
    print("  Done.")


def fig7_sample_paths():
    # forward SDE sample paths - GBM, CEV, Heston
    print("Generating Figure 7: Sample paths...")

    M_plot = 10
    N = 500

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    rng = np.random.default_rng(rng_seed)
    t_gbm, S_gbm = gbm_euler_maruyama(S0, r, sigma, T, N, M_plot, rng=rng)
    for j in range(M_plot):
        axes[0].plot(t_gbm, S_gbm[j, :], alpha=0.7, linewidth=0.8)
    axes[0].set_title('GBM ($\\sigma=0.20$)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$S_t$')

    rng = np.random.default_rng(rng_seed)
    t_cev, S_cev = cev_euler_maruyama(S0, r, 2.0, 0.5, T, N, M_plot, rng=rng)
    for j in range(M_plot):
        axes[1].plot(t_cev, S_cev[j, :], alpha=0.7, linewidth=0.8)
    axes[1].set_title('CEV ($\\gamma=0.5$)')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('$S_t$')

    rng = np.random.default_rng(rng_seed)
    t_hes, S_hes, v_hes = heston_euler_maruyama(
        S0, 0.04, r, 2.0, 0.04, 0.3, -0.7, T, N, M_plot, rng=rng)
    for j in range(M_plot):
        axes[2].plot(t_hes, S_hes[j, :], alpha=0.7, linewidth=0.8)
    axes[2].set_title('Heston (stochastic vol)')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('$S_t$')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig7_sample_paths.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig7_sample_paths.png'))
    plt.close()
    print("  Done.")


def fig8_bsde_nonlinear_comparison():
    # bsde prices under all 4 drivers
    print("Generating Figure 8: BSDE nonlinear comparison...")

    M = 50000
    N = 100
    spots = np.linspace(70, 130, 20)

    drivers = {
        'Linear (BS)': linear_driver(r),
        'Trans. costs ($\\kappa=0.01$)': transaction_cost_driver(r, 0.01),
        'Funding ($r_b=8\\%$)': funding_cost_driver(0.03, 0.08),
        'Uncertain vol': uncertain_vol_driver(r, 0.15, 0.30),
    }
    styles = ['b-', 'r--', 'g-.', 'm:']

    fig, ax = plt.subplots(figsize=(8, 5))

    for (label, driver), style in zip(drivers.items(), styles):
        prices = []
        for s in spots:
            rng = np.random.default_rng(rng_seed)
            t_fwd, X = gbm_euler_maruyama(s, r, sigma, T, N, M, rng=rng)
            Y, Z = solve_bsde(X, t_fwd, european_call_payoff(K),
                               driver, basis_degree=4)
            prices.append(np.mean(Y[:, 0]))
        ax.plot(spots, prices, style, linewidth=2, label=label)

    ax.set_xlabel('Spot Price $S_0$')
    ax.set_ylabel('Option Price')
    ax.set_title('BSDE Prices Under Different Market Frictions')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig8_bsde_nonlinear.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig8_bsde_nonlinear.png'))
    plt.close()
    print("  Done.")


if __name__ == '__main__':
    print("=" * 60)
    print("Generating figures for BSDE pricing paper")
    print("=" * 60)
    fig7_sample_paths()
    fig1_bsde_vs_pde()
    fig2_price_surfaces()
    fig3_bsde_convergence()
    fig4_sde_convergence()
    fig5_transaction_costs()
    fig6_uncertain_vol()
    fig8_bsde_nonlinear_comparison()
    print("=" * 60)
    print(f"All figures saved to {FIG_DIR}")
    print("=" * 60)
