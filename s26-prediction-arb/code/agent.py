import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from fees import kalshi_fee_dollars, net_edge_cross_platform
from orderbook import ArbitrageOpportunity


# ---- portfolio / trade data structures ----

@dataclass
class PortfolioState:
    cash: float = 100_000.0
    positions: Dict[str, int] = field(default_factory=dict)
    pnl_history: List[float] = field(default_factory=list)
    fees_paid: float = 0.0
    n_trades: int = 0
    errors: List[dict] = field(default_factory=list)

    @property
    def total_exposure(self):
        # sum of absolute position values
        return sum(abs(v) for v in self.positions.values())


@dataclass
class TradeDecision:
    action: str       # 'trade' or 'skip'
    size: int
    reasoning: str
    confidence: float


# ---- rule-based agent ----

class RuleBasedAgent:
    # deterministic, fee-aware, conservative

    def __init__(
        self,
        max_position: int = 100,
        max_exposure: int = 10_000,
        min_edge_pct: float = 0.02,
        kelly_frac: float = 0.25,
    ):
        self.max_position = max_position
        self.max_exposure = max_exposure
        self.min_edge_pct = min_edge_pct
        self.kelly_frac = kelly_frac

    def decide_trade(self, opp: ArbitrageOpportunity, state: PortfolioState) -> TradeDecision:
        # check if edge is worth it after fees
        if opp.edge < self.min_edge_pct:
            return TradeDecision('skip', 0, 'edge too thin', 0.0)

        # position limits
        current_pos = state.positions.get(opp.pair_id, 0)
        if abs(current_pos) >= self.max_position:
            return TradeDecision('skip', 0, 'max position reached', 0.0)

        # exposure check
        if state.total_exposure >= self.max_exposure:
            return TradeDecision('skip', 0, 'max exposure reached', 0.0)

        # size via fractional kelly
        kelly_size = self.kelly_frac * state.cash * opp.edge
        # can't exceed remaining position room or what's available
        room = self.max_position - abs(current_pos)
        max_from_exposure = max(self.max_exposure - state.total_exposure, 0)
        size = int(min(kelly_size, room, opp.max_size, max_from_exposure))

        if size <= 0:
            return TradeDecision('skip', 0, 'calculated size is zero', 0.0)

        return TradeDecision(
            'trade',
            size,
            f'edge={opp.edge:.4f} kelly_size={kelly_size:.0f} final={size}',
            min(opp.edge / self.min_edge_pct, 1.0),
        )

    def assess_risk(self, state: PortfolioState) -> dict:
        positions = state.positions
        largest = max(abs(v) for v in positions.values()) if positions else 0
        n_positions = len([v for v in positions.values() if v != 0])
        cash_pct = state.cash / (state.cash + state.total_exposure) if (state.cash + state.total_exposure) > 0 else 1.0

        return {
            'total_exposure': state.total_exposure,
            'largest_position': largest,
            'n_positions': n_positions,
            'cash_pct': cash_pct,
            'fees_paid': state.fees_paid,
        }


# ---- mock llm agent ----

class MockLLMAgent:
    # simulates a gpt-4 style agent that sometimes miscalculates fees
    # and occasionally trades on thin edges out of overconfidence

    def __init__(
        self,
        fee_error_rate: float = 0.12,
        overconf_factor: float = 1.2,
        seed: int = 42,
    ):
        self.fee_error_rate = fee_error_rate
        self.overconf_factor = overconf_factor
        self.rng = np.random.default_rng(seed)

        # same baseline params as rule-based
        self.max_position = 100
        self.max_exposure = 10_000
        self.min_edge_pct = 0.02
        self.kelly_frac = 0.25

    def decide_trade(self, opp: ArbitrageOpportunity, state: PortfolioState) -> TradeDecision:
        perceived_edge = opp.edge
        error_type = None

        # the llm agent messes up fees sometimes
        if self.rng.random() < self.fee_error_rate:
            # perturb edge by +/- 30% to simulate fee calc errors
            # (e.g., forgetting the ceiling, wrong coefficient, etc.)
            perturbation = self.rng.uniform(-0.3, 0.3)
            perceived_edge = opp.edge * (1.0 + perturbation)
            error_type = 'fee_miscalc'

        # overconfidence: sometimes trades when edge is marginal
        if error_type is None and perceived_edge < self.min_edge_pct:
            overconf_roll = self.rng.random()
            overconf_prob = self.overconf_factor - 1.0  # e.g. 0.2
            if overconf_roll < overconf_prob:
                # agent thinks it sees edge that isn't really there
                perceived_edge = self.min_edge_pct * 1.5
                error_type = 'overconfidence'

        # now make the decision based on (possibly wrong) perceived edge
        if perceived_edge < self.min_edge_pct:
            return TradeDecision('skip', 0, 'edge too thin (llm)', 0.0)

        current_pos = state.positions.get(opp.pair_id, 0)
        if abs(current_pos) >= self.max_position:
            return TradeDecision('skip', 0, 'max position (llm)', 0.0)

        if state.total_exposure >= self.max_exposure:
            return TradeDecision('skip', 0, 'max exposure (llm)', 0.0)

        kelly_size = self.kelly_frac * state.cash * perceived_edge
        room = self.max_position - abs(current_pos)
        max_from_exposure = max(self.max_exposure - state.total_exposure, 0)
        size = int(min(kelly_size, room, opp.max_size, max_from_exposure))

        if size <= 0:
            return TradeDecision('skip', 0, 'zero size (llm)', 0.0)

        reasoning = f'perceived_edge={perceived_edge:.4f} size={size}'
        if error_type:
            reasoning += f' [ERROR: {error_type}]'

        decision = TradeDecision(
            'trade',
            size,
            reasoning,
            min(perceived_edge / self.min_edge_pct, 1.0),
        )

        # log errors for later analysis
        if error_type:
            state.errors.append({
                'type': error_type,
                'pair_id': opp.pair_id,
                'true_edge': opp.edge,
                'perceived_edge': perceived_edge,
                'size': size,
            })

        return decision

    def assess_risk(self, state: PortfolioState) -> dict:
        # mostly correct but occasionally forgets a position
        positions = dict(state.positions)

        # simulate state tracking error: drop a random position ~10% of the time
        if len(positions) > 1 and self.rng.random() < 0.10:
            drop_key = self.rng.choice(list(positions.keys()))
            positions.pop(drop_key)

        largest = max(abs(v) for v in positions.values()) if positions else 0
        n_positions = len([v for v in positions.values() if v != 0])
        total_exp = sum(abs(v) for v in positions.values())
        cash_pct = state.cash / (state.cash + total_exp) if (state.cash + total_exp) > 0 else 1.0

        return {
            'total_exposure': total_exp,
            'largest_position': largest,
            'n_positions': n_positions,
            'cash_pct': cash_pct,
            'fees_paid': state.fees_paid,
        }


# ---- evaluation ----

def evaluate_agent_trades(
    decisions: List[TradeDecision],
    outcomes: List[float],
) -> dict:
    # outcomes[i] = actual pnl from trade i (positive = profit, negative = loss)
    # if decision was 'skip', outcome should be 0
    assert len(decisions) == len(outcomes)

    trades_taken = [(d, o) for d, o in zip(decisions, outcomes) if d.action == 'trade']
    n_trades = len(trades_taken)

    if n_trades == 0:
        return {
            'total_pnl': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_edge': 0.0,
            'n_trades': 0,
            'n_errors': 0,
        }

    pnls = [o for _, o in trades_taken]
    total_pnl = sum(pnls)

    # sharpe (annualized, assuming daily)
    pnl_arr = np.array(pnls)
    if pnl_arr.std() > 0:
        sharpe = (pnl_arr.mean() / pnl_arr.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # max drawdown from cumulative pnl
    cum = np.cumsum(pnl_arr)
    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum
    max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / n_trades

    avg_edge = np.mean([d.confidence for d, _ in trades_taken])

    # count errors (decisions that have ERROR in reasoning)
    n_errors = sum(1 for d, _ in trades_taken if 'ERROR' in d.reasoning)

    return {
        'total_pnl': total_pnl,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_edge': avg_edge,
        'n_trades': n_trades,
        'n_errors': n_errors,
    }


def compare_agents(rule_results: dict, llm_results: dict) -> dict:
    # side by side comparison, also prints a nice table
    comparison = {}
    all_keys = sorted(set(list(rule_results.keys()) + list(llm_results.keys())))

    print(f"\n{'metric':<20} {'rule-based':>12} {'mock-llm':>12} {'delta':>12}")
    print("-" * 58)

    for key in all_keys:
        rv = rule_results.get(key, 0)
        lv = llm_results.get(key, 0)

        comparison[key] = {
            'rule_based': rv,
            'mock_llm': lv,
            'delta': lv - rv if isinstance(rv, (int, float)) and isinstance(lv, (int, float)) else None,
        }

        # format for printing
        if isinstance(rv, float):
            print(f"{key:<20} {rv:>12.4f} {lv:>12.4f} {lv - rv:>+12.4f}")
        elif isinstance(rv, int):
            delta = lv - rv
            print(f"{key:<20} {rv:>12d} {lv:>12d} {delta:>+12d}")
        else:
            print(f"{key:<20} {str(rv):>12} {str(lv):>12} {'':>12}")

    print()
    return comparison
