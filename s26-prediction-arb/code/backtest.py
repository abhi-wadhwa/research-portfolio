import numpy as np
import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate_markets import simulate_backtest_data
from orderbook import OrderBook, detect_cross_platform_arb, scan_all_opportunities
from agent import RuleBasedAgent, MockLLMAgent, PortfolioState
from fees import kalshi_fee_dollars, polymarket_fee


# ---- trade execution sim ----

def execute_trade(opp, size, rng):
    # simulate fills with some slippage noise
    slippage = rng.uniform(0.0, 0.005)

    a_yes, b_no = opp.prices
    k_fee = kalshi_fee_dollars(size, a_yes)
    p_fee = polymarket_fee(size, b_no)

    gross_pnl = size * (1.0 - a_yes - b_no)
    net_pnl = gross_pnl - k_fee - p_fee - (slippage * size)

    return {
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'fees': k_fee + p_fee,
        'slippage': slippage * size,
        'fill_size': size,
    }


# ---- single agent backtest ----

def run_single_backtest(agent, backtest_data, min_edge=0.01, seed=42):
    rng = np.random.default_rng(seed)
    state = PortfolioState()
    daily_pnl = []
    error_log = []

    matched_pairs = backtest_data['matched_pairs']
    daily_books = backtest_data['daily_books']
    resolutions = backtest_data['resolutions']
    n_days = len(daily_books)

    for day in range(n_days):
        day_pnl = 0.0
        day_books = daily_books[day]

        # build orderbook objects for this day
        books_k = {}
        books_p = {}
        for pair_id in day_books:
            books_k[pair_id] = OrderBook.from_dict(day_books[pair_id]['kalshi'])
            books_p[pair_id] = OrderBook.from_dict(day_books[pair_id]['polymarket'])

        # scan for arb opportunities
        opps = scan_all_opportunities(matched_pairs, books_k, books_p, min_edge=min_edge)

        for opp in opps:
            decision = agent.decide_trade(opp, state)

            if decision.action == 'trade':
                pair_id = opp.pair_id
                result = execute_trade(opp, decision.size, rng)
                day_pnl += result['net_pnl']
                state.fees_paid += result['fees']
                state.n_trades += 1

                # update position tracking
                key = f"pair_{pair_id}"
                state.positions[key] = state.positions.get(key, 0) + decision.size

                # classify errors on losing trades
                if result['net_pnl'] < 0:
                    if result['slippage'] > result['fees']:
                        err_type = 'thin_market'
                    elif 'ERROR' in decision.reasoning:
                        if 'fee_miscalc' in decision.reasoning:
                            err_type = 'fee_calculation'
                        else:
                            err_type = 'match_error'
                    elif abs(result['net_pnl']) < 0.5:
                        err_type = 'bad_luck'
                    else:
                        err_type = 'book_misread'

                    error_log.append({
                        'day': day,
                        'pair_id': pair_id,
                        'type': err_type,
                        'pnl': result['net_pnl'],
                    })

        # handle resolutions for this day
        for pair_id, res in resolutions.items():
            if res['day'] == day:
                key = f"pair_{pair_id}"
                pos = state.positions.get(key, 0)
                if pos > 0:
                    if res['outcome'] == 'YES':
                        resolve_pnl = pos * (1.0 - backtest_data['price_history'][day, pair_id])
                    else:
                        resolve_pnl = -pos * backtest_data['price_history'][day, pair_id]
                    day_pnl += resolve_pnl
                    state.positions[key] = 0

        state.cash += day_pnl
        state.pnl_history.append(day_pnl)
        daily_pnl.append(day_pnl)

    # summary stats
    daily_pnl = np.array(daily_pnl)
    cum_pnl = np.cumsum(daily_pnl)
    total_pnl = float(cum_pnl[-1]) if len(cum_pnl) > 0 else 0.0

    if daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    if len(cum_pnl) > 0:
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = running_max - cum_pnl
        max_dd = float(drawdowns.max())
    else:
        max_dd = 0.0

    error_counts = {}
    for e in error_log:
        error_counts[e['type']] = error_counts.get(e['type'], 0) + 1

    return {
        'daily_pnl': daily_pnl,
        'cum_pnl': cum_pnl,
        'total_pnl': total_pnl,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'n_trades': state.n_trades,
        'fees_paid': state.fees_paid,
        'error_log': error_log,
        'error_counts': error_counts,
        'portfolio_state': state,
    }


# ---- comparison backtest across multiple seeds ----

def run_comparison_backtest(backtest_data, n_seeds=3, min_edge=0.01):
    rule_results = []
    llm_results = []

    for seed in range(n_seeds):
        print(f"  seed {seed + 1}/{n_seeds}...")

        rule_agent = RuleBasedAgent(min_edge_pct=min_edge)
        llm_agent = MockLLMAgent(seed=seed)

        r_result = run_single_backtest(rule_agent, backtest_data, min_edge=min_edge, seed=seed)
        l_result = run_single_backtest(llm_agent, backtest_data, min_edge=min_edge, seed=seed + 100)

        rule_results.append(r_result)
        llm_results.append(l_result)

    n_days = len(rule_results[0]['daily_pnl'])

    rule_cum_pnls = np.array([r['cum_pnl'] for r in rule_results])
    llm_cum_pnls = np.array([r['cum_pnl'] for r in llm_results])

    # average error counts across seeds
    rule_errors = {}
    llm_errors = {}
    for r in rule_results:
        for k, v in r['error_counts'].items():
            rule_errors[k] = rule_errors.get(k, 0) + v
    for r in llm_results:
        for k, v in r['error_counts'].items():
            llm_errors[k] = llm_errors.get(k, 0) + v

    for k in rule_errors:
        rule_errors[k] = rule_errors[k] / n_seeds
    for k in llm_errors:
        llm_errors[k] = llm_errors[k] / n_seeds

    return {
        'rule_cum_pnls': rule_cum_pnls,
        'llm_cum_pnls': llm_cum_pnls,
        'rule_mean_pnl': np.mean([r['total_pnl'] for r in rule_results]),
        'llm_mean_pnl': np.mean([r['total_pnl'] for r in llm_results]),
        'rule_mean_sharpe': np.mean([r['sharpe'] for r in rule_results]),
        'llm_mean_sharpe': np.mean([r['sharpe'] for r in llm_results]),
        'rule_errors': rule_errors,
        'llm_errors': llm_errors,
        'rule_results': rule_results,
        'llm_results': llm_results,
        'n_days': n_days,
    }


if __name__ == '__main__':
    print("generating backtest data...")
    bt_data = simulate_backtest_data(n_days=30, n_pairs=50, seed=42)

    print("running comparison...")
    results = run_comparison_backtest(bt_data, n_seeds=2, min_edge=0.005)

    print(f"\nrule-based: pnl=${results['rule_mean_pnl']:.2f}  sharpe={results['rule_mean_sharpe']:.2f}")
    print(f"mock-llm:   pnl=${results['llm_mean_pnl']:.2f}  sharpe={results['llm_mean_sharpe']:.2f}")
    print(f"rule errors: {results['rule_errors']}")
    print(f"llm errors:  {results['llm_errors']}")
