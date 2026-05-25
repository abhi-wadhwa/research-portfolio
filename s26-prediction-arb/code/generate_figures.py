import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})
sns.set_style("whitegrid")

from simulate_markets import generate_matching_benchmark, simulate_backtest_data, simulate_order_book
from matching import HybridMatcher, evaluate_matching, time_pipeline_stages
from fees import kalshi_fee, polymarket_fee, fee_impact_curve, net_edge_cross_platform
from orderbook import OrderBook, detect_cross_platform_arb, scan_all_opportunities
from agent import RuleBasedAgent, MockLLMAgent, PortfolioState
from backtest import run_comparison_backtest


def plot_matching_precision_recall(benchmark):
    # fig 1: PR curves for different pipeline configurations
    print("  generating PR curves for 5 pipeline variants...")

    configs = [
        ('Regex Only',          dict(use_regex=True,  use_embedding=False, use_fuzzy=False, use_llm=False)),
        ('Regex + Embedding',   dict(use_regex=True,  use_embedding=True,  use_fuzzy=False, use_llm=False)),
        ('Regex + Embed + Fuzz', dict(use_regex=True, use_embedding=True,  use_fuzzy=True,  use_llm=False)),
        ('Full Pipeline (+LLM)', dict(use_regex=True, use_embedding=True, use_fuzzy=True,  use_llm=True)),
        ('Embedding Only',      dict(use_regex=False, use_embedding=True,  use_fuzzy=False, use_llm=False)),
    ]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    linestyles = ['-', '--', '-.', '-', ':']
    thresholds = np.linspace(0.05, 0.95, 50)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for (name, kwargs), color, ls in zip(configs, colors, linestyles):
        matcher = HybridMatcher(**kwargs, threshold=0.5)
        result = evaluate_matching(benchmark, matcher, thresholds=thresholds)
        ax.plot(result['recalls'], result['precisions'], color=color, linestyle=ls,
                linewidth=1.8, label=f"{name} (F1={result['best_f1']:.2f})")

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Matching Pipeline: Precision vs. Recall')
    ax.legend(loc='lower left', fontsize=8)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])

    fig.savefig(os.path.join(FIGURES_DIR, 'matching_precision_recall.png'))
    plt.close(fig)
    print("  saved matching_precision_recall.png")


def plot_matching_f1_by_category(benchmark):
    # fig 2: grouped bar chart of f1 by category for each pipeline variant
    print("  computing per-category F1 scores...")

    configs = [
        ('Regex Only',          dict(use_regex=True,  use_embedding=False, use_fuzzy=False, use_llm=False)),
        ('Regex + Embed',       dict(use_regex=True,  use_embedding=True,  use_fuzzy=False, use_llm=False)),
        ('Regex + Embed + Fuzz', dict(use_regex=True, use_embedding=True,  use_fuzzy=True,  use_llm=False)),
        ('Full Pipeline',       dict(use_regex=True,  use_embedding=True,  use_fuzzy=True,  use_llm=True)),
    ]

    # normalize category names for display
    cat_display = {
        'nfl': 'Sports', 'nba': 'Sports', 'mlb': 'Sports',
        'crypto': 'Crypto', 'politics': 'Politics', 'economics': 'Economics',
    }
    display_cats = ['Sports', 'Crypto', 'Politics', 'Economics']

    # collect f1 per display category per pipeline
    all_f1s = {}  # {pipeline_name: {display_cat: f1}}
    for name, kwargs in configs:
        matcher = HybridMatcher(**kwargs, threshold=0.5)
        result = evaluate_matching(benchmark, matcher)
        cat_f1 = result['category_f1']

        # aggregate sports categories
        merged = {}
        for raw_cat, f1 in cat_f1.items():
            dc = cat_display.get(raw_cat, raw_cat.capitalize())
            if dc in merged:
                merged[dc] = max(merged[dc], f1)  # take the better one
            else:
                merged[dc] = f1

        all_f1s[name] = merged

    # build the grouped bar chart
    n_cats = len(display_cats)
    n_pipes = len(configs)
    x = np.arange(n_cats)
    width = 0.18
    palette = sns.color_palette("Set2", n_pipes)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (name, _) in enumerate(configs):
        vals = [all_f1s[name].get(c, 0) for c in display_cats]
        offset = (i - n_pipes / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=palette[i], edgecolor='white')
        # add value labels on top
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Category')
    ax.set_ylabel('F1 Score')
    ax.set_title('Matching F1 by Event Category')
    ax.set_xticks(x)
    ax.set_xticklabels(display_cats)
    ax.set_ylim([0, 1.15])
    ax.legend(fontsize=8)

    fig.savefig(os.path.join(FIGURES_DIR, 'matching_f1_by_category.png'))
    plt.close(fig)
    print("  saved matching_f1_by_category.png")


def plot_arbitrage_edge_distribution(backtest_data):
    # fig 3: histogram of gross vs net edge for detected arb opportunities
    print("  scanning all days for arb opportunities...")

    gross_edges = []
    net_edges = []

    daily_books = backtest_data['daily_books']
    matched_pairs = backtest_data['matched_pairs']

    for day in range(len(daily_books)):
        day_books = daily_books[day]
        books_k = {}
        books_p = {}
        for pair_id in day_books:
            books_k[pair_id] = OrderBook.from_dict(day_books[pair_id]['kalshi'])
            books_p[pair_id] = OrderBook.from_dict(day_books[pair_id]['polymarket'])

        # scan with very low min_edge to capture everything
        opps = scan_all_opportunities(matched_pairs, books_k, books_p, min_edge=-1.0)
        for opp in opps:
            a_yes, b_no = opp.prices
            gross = 1.0 - a_yes - b_no
            gross_edges.append(gross)
            net_edges.append(opp.edge)

    gross_edges = np.array(gross_edges)
    net_edges = np.array(net_edges)

    fig, ax = plt.subplots(figsize=(8, 5))

    # histograms
    bins = np.linspace(min(gross_edges.min(), net_edges.min()) - 0.01,
                       max(gross_edges.max(), net_edges.max()) + 0.01, 60)
    ax.hist(gross_edges, bins=bins, alpha=0.6, color='#1f77b4', label='Gross Edge', density=True)
    ax.hist(net_edges, bins=bins, alpha=0.6, color='#d62728', label='Net Edge (after fees)', density=True)

    # kde overlay
    from scipy.stats import gaussian_kde
    if len(gross_edges) > 5:
        kde_gross = gaussian_kde(gross_edges)
        kde_net = gaussian_kde(net_edges)
        xs = np.linspace(bins[0], bins[-1], 200)
        ax.plot(xs, kde_gross(xs), color='#1f77b4', linewidth=1.5, linestyle='--')
        ax.plot(xs, kde_net(xs), color='#d62728', linewidth=1.5, linestyle='--')

    # reference lines
    ax.axvline(0.0, color='black', linewidth=1.2, linestyle='-', alpha=0.7, label='Break-even')
    ax.axvline(0.02, color='green', linewidth=1.2, linestyle=':', alpha=0.7, label='Min edge (2%)')

    ax.set_xlabel('Edge per Contract ($)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Arbitrage Edge (Gross vs. Net)')
    ax.legend(fontsize=8)

    fig.savefig(os.path.join(FIGURES_DIR, 'arbitrage_edge_distribution.png'))
    plt.close(fig)
    print("  saved arbitrage_edge_distribution.png")


def plot_fee_impact():
    # fig 4: 2-panel fee analysis
    print("  computing fee curves...")

    prices = np.linspace(0.01, 0.99, 200)
    fee_data = fee_impact_curve(prices, contracts=100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # left panel: fee per contract vs price
    kalshi_per = fee_data['kalshi_fees'] / 100.0   # per contract
    poly_per = fee_data['poly_fees'] / 100.0       # per contract
    ax1.plot(prices, kalshi_per, color='#1f77b4', linewidth=1.8, label='Kalshi')
    ax1.plot(prices, poly_per, color='#d62728', linewidth=1.8, label='Polymarket')
    ax1.set_xlabel('Contract Price')
    ax1.set_ylabel('Fee per Contract ($)')
    ax1.set_title('Fee Structure: Kalshi vs. Polymarket')
    ax1.legend()

    # right panel: net edge as function of price disagreement
    disagreements = np.linspace(0.0, 0.20, 100)
    mid_price = 0.50  # representative mid-price

    gross_line = disagreements  # gross edge = |p_A - p_B| when one buys YES one buys NO
    net_line = []
    for d in disagreements:
        # kalshi YES at mid - d/2, poly NO at 1 - (mid + d/2) = 0.5 - d/2
        a_yes = mid_price - d / 2
        b_no = 1.0 - (mid_price + d / 2)
        # clamp
        a_yes = max(0.01, min(0.99, a_yes))
        b_no = max(0.01, min(0.99, b_no))
        ne = net_edge_cross_platform(a_yes, b_no, 100)
        net_line.append(ne)
    net_line = np.array(net_line)

    ax2.plot(disagreements, gross_line, color='gray', linewidth=1.5, linestyle='--', label='Gross Edge')
    ax2.plot(disagreements, net_line, color='#2ca02c', linewidth=1.8, linestyle='-', label='Net Edge')
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
    ax2.axhline(0.02, color='orange', linewidth=0.8, linestyle=':', alpha=0.7, label='Min profitable (2c)')
    ax2.set_xlabel('Price Disagreement |p_A - p_B|')
    ax2.set_ylabel('Edge per Contract ($)')
    ax2.set_title('Net Edge vs. Cross-Platform Disagreement (mid=0.50)')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fee_impact_analysis.png'))
    plt.close(fig)
    print("  saved fee_impact_analysis.png")


def plot_pnl_curves(bt_results):
    # fig 5: cumulative pnl for rule-based vs llm agent
    print("  plotting pnl curves...")

    rule_cum = bt_results['rule_cum_pnls']  # shape (n_seeds, n_days)
    llm_cum = bt_results['llm_cum_pnls']
    n_days = bt_results['n_days']
    days = np.arange(1, n_days + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    # if multiple seeds, show mean +/- std shaded region
    if rule_cum.shape[0] > 1:
        rule_mean = rule_cum.mean(axis=0)
        rule_std = rule_cum.std(axis=0)
        llm_mean = llm_cum.mean(axis=0)
        llm_std = llm_cum.std(axis=0)

        ax.plot(days, rule_mean, color='#1f77b4', linewidth=1.8,
                label=f"Rule-based (PnL=${bt_results['rule_mean_pnl']:.0f}, Sharpe={bt_results['rule_mean_sharpe']:.2f})")
        ax.fill_between(days, rule_mean - rule_std, rule_mean + rule_std,
                         color='#1f77b4', alpha=0.15)

        ax.plot(days, llm_mean, color='#d62728', linewidth=1.8,
                label=f"LLM Agent (PnL=${bt_results['llm_mean_pnl']:.0f}, Sharpe={bt_results['llm_mean_sharpe']:.2f})")
        ax.fill_between(days, llm_mean - llm_std, llm_mean + llm_std,
                         color='#d62728', alpha=0.15)
    else:
        ax.plot(days, rule_cum[0], color='#1f77b4', linewidth=1.8,
                label=f"Rule-based (PnL=${bt_results['rule_mean_pnl']:.0f}, Sharpe={bt_results['rule_mean_sharpe']:.2f})")
        ax.plot(days, llm_cum[0], color='#d62728', linewidth=1.8,
                label=f"LLM Agent (PnL=${bt_results['llm_mean_pnl']:.0f}, Sharpe={bt_results['llm_mean_sharpe']:.2f})")

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='-', alpha=0.5)
    ax.set_xlabel('Day')
    ax.set_ylabel('Cumulative PnL ($)')
    ax.set_title('90-Day Paper Trading: Rule-Based vs. LLM Agent')
    ax.legend(loc='upper left', fontsize=8)

    fig.savefig(os.path.join(FIGURES_DIR, 'pnl_curves.png'))
    plt.close(fig)
    print("  saved pnl_curves.png")


def plot_orderbook_visualization():
    # fig 6: side-by-side order books showing a cross-platform arb
    print("  building representative order book visualization...")

    rng = np.random.default_rng(7)

    # create two books with a price gap that creates an arb
    k_book_raw = simulate_order_book(0.47, depth=8, spread_cents=2, vol_mean=40, rng=rng)
    p_book_raw = simulate_order_book(0.52, depth=8, spread_cents=2, vol_mean=35, rng=rng)

    k_book = OrderBook.from_dict(k_book_raw)
    p_book = OrderBook.from_dict(p_book_raw)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # helper to draw horizontal book
    def draw_book(ax, book, title):
        n_levels = min(8, len(book.bid_prices))

        # bids on the left (green), asks on the right (red)
        bid_prices = book.bid_prices[:n_levels]
        bid_vols = book.bid_volumes[:n_levels]
        ask_prices = book.ask_prices[:n_levels]
        ask_vols = book.ask_volumes[:n_levels]

        y_positions = np.arange(n_levels)

        # bids: extend left from center
        ax.barh(y_positions, -bid_vols, height=0.6, color='#2ca02c', alpha=0.7,
                edgecolor='white', label='Bids')
        # asks: extend right from center
        ax.barh(y_positions, ask_vols, height=0.6, color='#d62728', alpha=0.7,
                edgecolor='white', label='Asks')

        # label price levels
        for i in range(n_levels):
            ax.text(-bid_vols[i] - 3, i, f'${bid_prices[i]:.2f}', ha='right', va='center', fontsize=7, color='#2ca02c')
            ax.text(ask_vols[i] + 3, i, f'${ask_prices[i]:.2f}', ha='left', va='center', fontsize=7, color='#d62728')

        ax.set_yticks(y_positions)
        ax.set_yticklabels([f'L{i+1}' for i in range(n_levels)], fontsize=8)
        ax.axvline(0, color='gray', linewidth=0.8)
        ax.set_xlabel('Volume')
        ax.set_title(title)
        ax.legend(fontsize=8, loc='lower right')

    draw_book(ax1, k_book, 'Kalshi Order Book')
    draw_book(ax2, p_book, 'Polymarket Order Book')

    # annotate the arb opportunity
    k_ask = k_book.best_ask
    p_bid = p_book.best_bid
    gross_edge = (1.0 - k_ask - (1.0 - p_bid))  # = p_bid - k_ask

    ax1.annotate(f'Buy YES @ ${k_ask:.2f}',
                 xy=(k_book.ask_volumes[0] * 0.5, 0), fontsize=9, fontweight='bold',
                 color='darkred',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    ax2.annotate(f'Sell YES @ ${p_bid:.2f}\n(= Buy NO @ ${1.0 - p_bid:.2f})',
                 xy=(-p_book.bid_volumes[0] * 0.5, 0), fontsize=9, fontweight='bold',
                 color='darkgreen',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    # add gross edge annotation between panels
    fig.text(0.5, 0.02, f'Gross Edge = ${gross_edge:.2f}/contract',
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='orange'))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(FIGURES_DIR, 'orderbook_visualization.png'))
    plt.close(fig)
    print("  saved orderbook_visualization.png")


def plot_failure_modes(bt_results):
    # fig 7: stacked bar chart of error types for rule-based vs llm agent
    print("  building failure mode breakdown...")

    error_types = ['fee_calculation', 'book_misread', 'match_error', 'thin_market', 'bad_luck']
    error_colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#9467bd', '#7f7f7f']
    error_labels = ['Fee Calculation', 'Book Misread', 'Match Error', 'Thin Market', 'Bad Luck']

    rule_errors = bt_results['rule_errors']
    llm_errors = bt_results['llm_errors']

    rule_counts = [rule_errors.get(t, 0) for t in error_types]
    llm_counts = [llm_errors.get(t, 0) for t in error_types]

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.array([0, 1])
    bar_width = 0.5

    # stacked bars
    rule_bottom = 0
    llm_bottom = 0
    for i, (etype, color, label) in enumerate(zip(error_types, error_colors, error_labels)):
        rc = rule_counts[i]
        lc = llm_counts[i]
        ax.bar(0, rc, bar_width, bottom=rule_bottom, color=color, edgecolor='white', label=label if i < len(error_types) else None)
        ax.bar(1, lc, bar_width, bottom=llm_bottom, color=color, edgecolor='white')
        rule_bottom += rc
        llm_bottom += lc

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Rule-Based', 'LLM Agent'])
    ax.set_ylabel('Number of Errors (avg across seeds)')
    ax.set_title('Failure Mode Breakdown')
    ax.legend(loc='upper right', fontsize=8)

    fig.savefig(os.path.join(FIGURES_DIR, 'failure_modes.png'))
    plt.close(fig)
    print("  saved failure_modes.png")


def plot_matching_latency(benchmark):
    # fig 8: bar chart of per-stage latency
    print("  timing pipeline stages on 100 sample pairs...")

    timings = time_pipeline_stages(benchmark, n_sample=100)

    stages = ['Regex', 'SBERT Embed', 'Rapidfuzz', 'LLM Verify', 'Full Pipeline']
    means = [timings[s]['mean'] for s in stages]
    stds = [timings[s]['std'] for s in stages]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(stages, means, yerr=stds, capsize=4, color=colors,
                  edgecolor='white', alpha=0.85)

    ax.set_yscale('log')
    ax.set_ylabel('Latency per Pair (ms, log scale)')
    ax.set_title('Matching Pipeline Latency by Stage')
    ax.set_xticklabels(stages, rotation=15, ha='right')

    # value labels
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                f'{m:.2f}ms', ha='center', va='bottom', fontsize=8)

    fig.savefig(os.path.join(FIGURES_DIR, 'matching_latency.png'))
    plt.close(fig)
    print("  saved matching_latency.png")


def main():
    print("generating matching benchmark data...")
    benchmark = generate_matching_benchmark(n_pairs=500, seed=42)
    print(f"  {len(benchmark)} total pairs")

    print("generating backtest market data...")
    bt_data = simulate_backtest_data(n_days=90, n_pairs=100, seed=42)
    print(f"  {len(bt_data['matched_pairs'])} pairs, {len(bt_data['daily_books'])} days")

    print("running comparison backtest...")
    bt_results = run_comparison_backtest(bt_data, n_seeds=3, min_edge=0.01)
    print(f"  rule pnl=${bt_results['rule_mean_pnl']:.0f}  llm pnl=${bt_results['llm_mean_pnl']:.0f}")

    print("\nfig 1/8: matching PR curves...")
    plot_matching_precision_recall(benchmark)

    print("fig 2/8: matching F1 by category...")
    plot_matching_f1_by_category(benchmark)

    print("fig 3/8: arbitrage edge distribution...")
    plot_arbitrage_edge_distribution(bt_data)

    print("fig 4/8: fee impact analysis...")
    plot_fee_impact()

    print("fig 5/8: pnl curves...")
    plot_pnl_curves(bt_results)

    print("fig 6/8: order book visualization...")
    plot_orderbook_visualization()

    print("fig 7/8: failure modes...")
    plot_failure_modes(bt_results)

    print("fig 8/8: matching latency...")
    plot_matching_latency(benchmark)

    print("\nall figures saved to", FIGURES_DIR)


if __name__ == '__main__':
    main()
