import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# fig9-fig16 for expanded paper

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)
plt.rcParams.update({
    'figure.figsize': (7, 5),
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.family': 'serif',
})


def fig9_deep_bsde_training():
    # training loss curves for each benchmark
    print("Generating Figure 9: Deep BSDE training curves...")
    from deep_bsde_solver import run_benchmark

    dim = 10
    n_epochs = 800
    sigma_val = 1.0
    benchmarks = {
        'Black-Scholes': ('black_scholes', 0.0),
        'Allen-Cahn': ('allen_cahn', 0.0),
        'HJB': ('hjb', 0.0),
        'Bergman': ('bergman', 1.0),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (label, (name, x0_val)) in enumerate(benchmarks.items()):
        print(f"  Training {label} (d={dim})...")
        model, losses, y0_hist, exact = run_benchmark(
            name, dim=dim, T=1.0, N=20, sigma_val=sigma_val,
            n_epochs=n_epochs, batch_size=256, lr=5e-3,
            hidden_dim=64, num_layers=2, x0_value=x0_val
        )
        ax = axes[idx]
        clean_losses = [l for l in losses if not np.isnan(l) and l > 0]
        if clean_losses:
            ax.semilogy(clean_losses, linewidth=1.5)
        else:
            ax.text(0.5, 0.5, 'Training diverged', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Terminal Loss $\\| Y_N - g(X_N) \\|^2$')
        ax.set_title(f'{label} ($d={dim}$)')
        y0_val = model.y0.item()
        if not np.isnan(y0_val) and exact is not None:
            ax.text(0.95, 0.95, f'$Y_0$={y0_val:.3f}\nExact={exact:.3f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        elif not np.isnan(y0_val):
            ax.text(0.95, 0.95, f'$Y_0$={y0_val:.4f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig9_deep_bsde_training.png'))
    plt.savefig(os.path.join(FIG_DIR, 'fig9_deep_bsde_training.pdf'))
    plt.close()
    print("  Done.")


def fig10_deep_bsde_convergence():
    # rel error vs dimension for BS benchmark
    print("Generating Figure 10: Deep BSDE dimension scaling...")
    from deep_bsde_solver import dimension_scaling

    dims = [5, 10, 20, 50]
    sigma_val = 1.0

    results = dimension_scaling(
        'black_scholes', dims, T=1.0, N=20, sigma_val=sigma_val,
        n_epochs=800, batch_size=256, lr=5e-3, hidden_dim=64, x0_value=0.0
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dims_plot = [r['dim'] for r in results]
    y0_vals = [r['y0'] for r in results]
    exact_vals = [r['exact'] for r in results]

    axes[0].plot(dims_plot, exact_vals, 'bs-', linewidth=2, markersize=8,
                 label='Exact')
    axes[0].plot(dims_plot, y0_vals, 'ro--', linewidth=2, markersize=8,
                 label='Deep BSDE')
    axes[0].set_xlabel('Dimension $d$')
    axes[0].set_ylabel('$Y_0$')
    axes[0].set_title('Deep BSDE vs Exact Solution')
    axes[0].legend()

    rel_errors = [r['rel_error'] if r['rel_error'] is not None else 0 for r in results]
    axes[1].semilogy(dims_plot, rel_errors, 'ko-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Dimension $d$')
    axes[1].set_ylabel('Relative Error $|Y_0 - u(0,x)|/|u(0,x)|$')
    axes[1].set_title('Deep BSDE Relative Error vs Dimension')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig10_deep_bsde_convergence.png'))
    plt.savefig(os.path.join(FIG_DIR, 'fig10_deep_bsde_convergence.pdf'))
    plt.close()
    print("  Done.")


def fig11_deep_bsde_solutions():
    # bar chart Y_0 vs reference
    print("Generating Figure 11: Deep BSDE solutions summary...")
    from deep_bsde_solver import run_benchmark

    dim = 10
    sigma_val = 1.0
    n_epochs = 800

    benchmarks = [
        ('Black-Scholes', 'black_scholes', 0.0),
        ('Allen-Cahn', 'allen_cahn', 0.0),
        ('HJB', 'hjb', 0.0),
        ('Bergman', 'bergman', 1.0),
    ]

    names = []
    y0_deep = []
    y0_exact = []
    has_exact = []
    all_losses = {}

    for label, name, x0_val in benchmarks:
        print(f"  Running {label}...")
        model, losses, _, exact = run_benchmark(
            name, dim=dim, T=1.0, N=20, sigma_val=sigma_val,
            n_epochs=n_epochs, batch_size=256, lr=5e-3,
            hidden_dim=64, num_layers=2, x0_value=x0_val
        )
        y0_val = model.y0.item()
        names.append(label)
        y0_deep.append(y0_val if not np.isnan(y0_val) else 0.0)
        y0_exact.append(exact if exact is not None else (y0_val if not np.isnan(y0_val) else 0.0))
        has_exact.append(exact is not None)
        all_losses[label] = losses

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_pos = np.arange(len(names))
    width = 0.35
    bars1 = axes[0].bar(x_pos - width / 2, y0_deep, width, label='Deep BSDE',
                         color='steelblue', edgecolor='black')
    for i_bar in range(len(names)):
        alpha_val = 0.9 if has_exact[i_bar] else 0.3
        axes[0].bar(x_pos[i_bar] + width / 2, y0_exact[i_bar], width,
                     color='coral', edgecolor='black', alpha=alpha_val,
                     label='Reference' if i_bar == 0 else None)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(names, fontsize=9)
    axes[0].set_ylabel('$Y_0$')
    axes[0].set_title(f'Deep BSDE Solutions ($d={dim}$)')
    axes[0].legend()

    for label, losses in all_losses.items():
        clean = [l for l in losses if not np.isnan(l) and l > 0]
        if clean:
            axes[1].semilogy(clean, linewidth=1.5, label=label)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Convergence Comparison')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig11_deep_bsde_solutions.png'))
    plt.savefig(os.path.join(FIG_DIR, 'fig11_deep_bsde_solutions.pdf'))
    plt.close()
    print("  Done.")


def fig12_cva_exposures():
    # EE, EPE profiles
    print("Generating Figure 12: CVA exposure profiles...")
    from cva_bsde import run_cva_analysis

    results = run_cva_analysis(M=20000, N=100, T=10.0, n_swaps=7)
    t = results['t_grid']
    EE = results['EE']
    V_paths = results['V_paths']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    NE = -np.mean(np.minimum(V_paths, 0.0), axis=0)
    axes[0].plot(t, EE, 'b-', linewidth=2, label='Expected Exposure (EE)')
    axes[0].plot(t, NE, 'r--', linewidth=2, label='Expected Negative Exposure')
    axes[0].fill_between(t, 0, EE, alpha=0.15, color='blue')
    axes[0].axhline(y=results['EPE'], color='green', linestyle=':', linewidth=1.5,
                     label=f'EPE = {results["EPE"]:,.0f}')
    axes[0].set_xlabel('Time (years)')
    axes[0].set_ylabel('Exposure')
    axes[0].set_title('Expected Exposure Profile')
    axes[0].legend(fontsize=9)

    q95 = np.percentile(np.maximum(V_paths, 0), 95, axis=0)
    q05 = np.percentile(np.maximum(V_paths, 0), 5, axis=0)
    axes[1].plot(t, EE, 'b-', linewidth=2, label='EE (mean)')
    axes[1].plot(t, q95, 'r--', linewidth=1.5, label='95th percentile')
    axes[1].fill_between(t, q05, q95, alpha=0.15, color='orange',
                          label='5th-95th percentile')
    axes[1].set_xlabel('Time (years)')
    axes[1].set_ylabel('Positive Exposure')
    axes[1].set_title('Exposure Distribution')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig12_cva_exposures.png'))
    plt.savefig(os.path.join(FIG_DIR, 'fig12_cva_exposures.pdf'))
    plt.close()
    print("  Done.")
    return results


def fig13_cva_bsde(cva_results=None):
    # CVA: BSDE vs MC
    print("Generating Figure 13: CVA BSDE vs MC...")
    if cva_results is None:
        from cva_bsde import run_cva_analysis
        cva_results = run_cva_analysis(M=20000, N=100, T=10.0, n_swaps=7)

    t = cva_results['t_grid']
    cva_mc_profile = cva_results['cva_mc_profile']
    Y_bsde = cva_results['Y_bsde']
    V_riskfree = cva_results['V_riskfree']

    bsde_diff = np.mean(V_riskfree, axis=0) - np.mean(Y_bsde, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(t, cva_mc_profile, 'b-', linewidth=2, label='Monte Carlo CVA')
    axes[0].plot(t, bsde_diff, 'r--', linewidth=2, label='BSDE CVA')
    axes[0].set_xlabel('Time (years)')
    axes[0].set_ylabel('Cumulative CVA')
    axes[0].set_title('CVA: BSDE vs Monte Carlo')
    axes[0].legend()
    axes[0].text(0.05, 0.95,
                  f'MC CVA = {cva_results["cva_mc"]:,.0f}\n'
                  f'BSDE CVA = {cva_results["cva_bsde"]:,.0f}',
                  transform=axes[0].transAxes, va='top', fontsize=9,
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[1].plot(t, np.mean(V_riskfree, axis=0), 'b-', linewidth=2,
                  label='Risk-free value')
    axes[1].plot(t, np.mean(Y_bsde, axis=0), 'r--', linewidth=2,
                  label='CVA-adjusted (BSDE)')
    axes[1].fill_between(t, np.mean(Y_bsde, axis=0), np.mean(V_riskfree, axis=0),
                          alpha=0.2, color='red', label='CVA adjustment')
    axes[1].set_xlabel('Time (years)')
    axes[1].set_ylabel('Portfolio Value')
    axes[1].set_title('Risk-Free vs CVA-Adjusted Value')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig13_cva_bsde.png'))
    plt.savefig(os.path.join(FIG_DIR, 'fig13_cva_bsde.pdf'))
    plt.close()
    print("  Done.")


def fig14_g_expectation():
    # g-exp vs cond exp
    print("Generating Figure 14: g-expectation analysis...")
    from g_expectation import run_g_expectation_analysis

    results = run_g_expectation_analysis(M=30000, N=100, seed=42)
    t = results['t_grid']
    g_res = results['g_results']
    gammas = results['gammas']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = sns.color_palette('coolwarm', len(gammas))
    for gamma, color in zip(gammas, colors):
        Y = g_res[gamma]['Y']
        Y_mean = np.mean(Y, axis=0)
        label = f'$\\gamma = {gamma}$'
        if gamma == 0:
            label = '$\\gamma = 0$ (linear $\\mathbb{E}$)'
        axes[0].plot(t, Y_mean, color=color, linewidth=2, label=label)

    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$\\mathcal{E}^g_t[\\xi]$')
    axes[0].set_title('g-Expectation vs Time')
    axes[0].legend(fontsize=8)

    y0_vals = [g_res[g]['Y0'] for g in gammas]
    axes[1].plot(gammas, y0_vals, 'ko-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Risk Aversion $\\gamma$')
    axes[1].set_ylabel('$\\mathcal{E}^g_0[\\xi]$')
    axes[1].set_title('Initial g-Expectation vs Risk Aversion')

    entropic = results['entropic_results']
    for theta, rho in entropic.items():
        axes[1].axhline(y=-rho, color='gray', linestyle=':', alpha=0.5)
        axes[1].annotate(f'Entropic ($\\theta$={theta})',
                          xy=(gammas[-1] * 0.6, -rho),
                          fontsize=7, color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig14_g_expectation.png'))
    plt.savefig(os.path.join(FIG_DIR, 'fig14_g_expectation.pdf'))
    plt.close()
    print("  Done.")
    return results


def fig15_time_consistency(g_results=None):
    # time-consistency check for g-exp
    print("Generating Figure 15: Time-consistency verification...")

    if g_results is None:
        from g_expectation import run_g_expectation_analysis
        g_results = run_g_expectation_analysis(M=30000, N=100, seed=42)

    tc_points = g_results['tc_time_points']
    gamma_tc = g_results['tc_gamma']
    tc_res = g_results['tc_results']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    times = [p['time'] for p in tc_points]
    directs = [p['direct'] for p in tc_points]
    composeds = [p['composed'] for p in tc_points]
    errors = [p['error'] for p in tc_points]

    axes[0].plot(times, directs, 'bo-', linewidth=2, markersize=6,
                  label='$\\mathcal{E}^g_0[\\xi]$ (direct)')
    axes[0].plot(times, composeds, 'rs--', linewidth=2, markersize=6,
                  label='$\\mathcal{E}^g_0[\\mathcal{E}^g_t[\\xi]]$ (composed)')
    axes[0].set_xlabel('Intermediate Time $t$')
    axes[0].set_ylabel('$\\mathcal{E}^g_0[\\cdot]$')
    axes[0].set_title(f'Time-Consistency ($\\gamma = {gamma_tc}$)')
    axes[0].legend(fontsize=9)

    axes[1].semilogy(times, errors, 'ko-', linewidth=2, markersize=6)
    axes[1].set_xlabel('Intermediate Time $t$')
    axes[1].set_ylabel('$|\\mathcal{E}^g_0[\\xi] - \\mathcal{E}^g_0[\\mathcal{E}^g_t[\\xi]]|$')
    axes[1].set_title('Time-Consistency Error')

    for gamma, data in tc_res.items():
        axes[1].axhline(y=data['error'], color='gray', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig15_time_consistency.png'))
    plt.savefig(os.path.join(FIG_DIR, 'fig15_time_consistency.pdf'))
    plt.close()
    print("  Done.")


def fig16_ablation():
    # ablation: width, depth, lr
    print("Generating Figure 16: Ablation study...")
    from deep_bsde_solver import ablation_study

    results = ablation_study(
        name='black_scholes', dim=10, T=1.0, N=20,
        sigma_val=np.sqrt(2.0), n_epochs=300, batch_size=256, x0_value=0.0
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    hd_data = results['hidden_dim']
    hd_vals = [d['value'] for d in hd_data]
    hd_errors = [d['rel_error'] if d['rel_error'] is not None else 0 for d in hd_data]
    ax = axes[0]
    ax.bar(range(len(hd_vals)), hd_errors, tick_label=[str(v) for v in hd_vals],
           color='steelblue', edgecolor='black')
    ax.set_xlabel('Hidden Dimension')
    ax.set_ylabel('Relative Error')
    ax.set_title('Effect of Network Width')

    nl_data = results['num_layers']
    nl_vals = [d['value'] for d in nl_data]
    nl_errors = [d['rel_error'] if d['rel_error'] is not None else 0 for d in nl_data]
    ax = axes[1]
    ax.bar(range(len(nl_vals)), nl_errors, tick_label=[str(v) for v in nl_vals],
           color='coral', edgecolor='black')
    ax.set_xlabel('Number of Hidden Layers')
    ax.set_ylabel('Relative Error')
    ax.set_title('Effect of Network Depth')

    lr_data = results['lr']
    lr_vals = [d['value'] for d in lr_data]
    lr_errors = [d['rel_error'] if d['rel_error'] is not None else 0 for d in lr_data]
    ax = axes[2]
    ax.bar(range(len(lr_vals)), lr_errors,
           tick_label=[f'{v:.0e}' for v in lr_vals],
           color='seagreen', edgecolor='black')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Relative Error')
    ax.set_title('Effect of Learning Rate')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig16_ablation_deep_bsde.png'))
    plt.savefig(os.path.join(FIG_DIR, 'fig16_ablation_deep_bsde.pdf'))
    plt.close()
    print("  Done.")


if __name__ == '__main__':
    print("=" * 60)
    print("Generating new figures for expanded BSDE paper")
    print("=" * 60)

    fig9_deep_bsde_training()
    fig10_deep_bsde_convergence()
    fig11_deep_bsde_solutions()

    cva_results = fig12_cva_exposures()
    fig13_cva_bsde(cva_results)

    g_results = fig14_g_expectation()
    fig15_time_consistency(g_results)

    fig16_ablation()

    print("=" * 60)
    print(f"All new figures saved to {FIG_DIR}")
    print("=" * 60)
