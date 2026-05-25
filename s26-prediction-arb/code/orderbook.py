import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from fees import kalshi_fee_dollars, polymarket_fee, net_edge_cross_platform


class OrderBook:
    # basic order book repr, good enough for sim data
    def __init__(self, bid_prices, bid_volumes, ask_prices, ask_volumes, timestamp=0):
        self.bid_prices = np.asarray(bid_prices, dtype=float)   # descending
        self.bid_volumes = np.asarray(bid_volumes, dtype=int)
        self.ask_prices = np.asarray(ask_prices, dtype=float)   # ascending
        self.ask_volumes = np.asarray(ask_volumes, dtype=int)
        self.timestamp = timestamp

    @property
    def best_bid(self):
        if len(self.bid_prices) == 0:
            return None
        return float(self.bid_prices[0])

    @property
    def best_ask(self):
        if len(self.ask_prices) == 0:
            return None
        return float(self.ask_prices[0])

    @property
    def spread(self):
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def mid_price(self):
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    def available_volume(self, side, up_to_price):
        # total volume on given side up to a price limit
        if side == 'bid':
            # bids: count levels where price >= up_to_price
            mask = self.bid_prices >= up_to_price
            return int(np.sum(self.bid_volumes[mask]))
        elif side == 'ask':
            # asks: count levels where price <= up_to_price
            mask = self.ask_prices <= up_to_price
            return int(np.sum(self.ask_volumes[mask]))
        else:
            raise ValueError(f"side must be 'bid' or 'ask', got {side}")

    def cost_to_fill(self, side, quantity):
        # walk the book til we fill
        if side == 'ask':
            prices = self.ask_prices
            volumes = self.ask_volumes
        elif side == 'bid':
            prices = self.bid_prices
            volumes = self.bid_volumes
        else:
            raise ValueError(f"side must be 'bid' or 'ask', got {side}")

        filled = 0
        total_cost = 0.0

        for px, vol in zip(prices, volumes):
            can_fill = min(vol, quantity - filled)
            total_cost += can_fill * px
            filled += can_fill
            if filled >= quantity:
                break

        # return avg fill price and actual filled qty
        avg_price = total_cost / filled if filled > 0 else 0.0
        return avg_price, int(filled)

    @classmethod
    def from_dict(cls, d):
        # construct from the dict returned by simulate_order_book
        return cls(
            bid_prices=d['bid_prices'],
            bid_volumes=d['bid_volumes'],
            ask_prices=d['ask_prices'],
            ask_volumes=d['ask_volumes'],
        )


@dataclass
class ArbitrageOpportunity:
    type: str           # 'cross_platform', 'implied', 'monotonicity'
    pair_id: int
    edge: float         # net edge per contract after fees
    max_size: int       # max fillable size
    prices: tuple       # (leg1_price, leg2_price)
    timestamp: int = 0


def detect_cross_platform_arb(book_k, book_p, pair_id, poly_coeff=0.02, min_edge=0.01):
    # 4 possible trade directions
    opps = []

    # direction 1: buy YES kalshi (lift ask) + buy NO poly (lift ask)
    # YES kalshi costs ask_k, NO poly costs (1 - bid_p) effectively...
    # actually NO on poly means buying the NO token at poly's ask for the NO side
    # but we only have one book per market, so:
    # book_k = YES book on kalshi, book_p = YES book on polymarket
    # buying NO on poly = selling YES on poly = hitting poly's bid
    # buying YES on kalshi = lifting kalshi's ask

    if book_k.best_ask is not None and book_p.best_bid is not None:
        a_yes = book_k.best_ask     # cost of YES on kalshi
        b_no = 1.0 - book_p.best_bid  # cost of NO on poly (= 1 - YES bid)
        size = min(
            book_k.ask_volumes[0] if len(book_k.ask_volumes) > 0 else 0,
            book_p.bid_volumes[0] if len(book_p.bid_volumes) > 0 else 0,
        )
        if size > 0:
            edge = net_edge_cross_platform(a_yes, b_no, size, poly_coeff)
            if edge >= min_edge:
                opps.append(ArbitrageOpportunity(
                    type='cross_platform',
                    pair_id=pair_id,
                    edge=edge,
                    max_size=size,
                    prices=(a_yes, b_no),
                    timestamp=book_k.timestamp,
                ))

    # direction 2: buy NO kalshi (sell YES = hit bid) + buy YES poly (lift ask)
    if book_k.best_bid is not None and book_p.best_ask is not None:
        a_no = 1.0 - book_k.best_bid   # cost of NO on kalshi
        b_yes = book_p.best_ask          # cost of YES on poly
        size = min(
            book_k.bid_volumes[0] if len(book_k.bid_volumes) > 0 else 0,
            book_p.ask_volumes[0] if len(book_p.ask_volumes) > 0 else 0,
        )
        if size > 0:
            # same formula, just swap which platform is YES vs NO
            gross = 1.0 - b_yes - a_no
            k_fee = kalshi_fee_dollars(size, 1.0 - a_no) / size  # kalshi NO price
            p_fee = polymarket_fee(1, b_yes, poly_coeff)
            edge = gross - k_fee - p_fee
            if edge >= min_edge:
                opps.append(ArbitrageOpportunity(
                    type='cross_platform',
                    pair_id=pair_id,
                    edge=edge,
                    max_size=size,
                    prices=(a_no, b_yes),
                    timestamp=book_k.timestamp,
                ))

    return opps


def detect_implied_arb(book_yes, book_no, pair_id, min_edge=0.005):
    # same-platform: if we can buy YES and NO for less than $1 total
    opps = []

    if book_yes.best_ask is None or book_no.best_ask is None:
        return opps

    p_yes = book_yes.best_ask
    p_no = book_no.best_ask

    # gross edge before fees
    gross = 1.0 - p_yes - p_no
    if gross <= 0:
        return opps

    size = min(
        book_yes.ask_volumes[0] if len(book_yes.ask_volumes) > 0 else 0,
        book_no.ask_volumes[0] if len(book_no.ask_volumes) > 0 else 0,
    )
    if size == 0:
        return opps

    # subtract kalshi fees on both legs
    k_fee_yes = kalshi_fee_dollars(size, p_yes) / size
    k_fee_no = kalshi_fee_dollars(size, p_no) / size
    edge = gross - k_fee_yes - k_fee_no

    if edge >= min_edge:
        opps.append(ArbitrageOpportunity(
            type='implied',
            pair_id=pair_id,
            edge=edge,
            max_size=size,
            prices=(p_yes, p_no),
            timestamp=book_yes.timestamp,
        ))

    return opps


def detect_monotonicity_violations(books_by_threshold, pair_id):
    # for strike-ladder markets (crypto, econ) prices should decrease with threshold
    # e.g., P(BTC > 50k) >= P(BTC > 55k) >= P(BTC > 60k)
    opps = []

    thresholds = sorted(books_by_threshold.keys())
    if len(thresholds) < 2:
        return opps

    for i in range(len(thresholds) - 1):
        t_lo = thresholds[i]
        t_hi = thresholds[i + 1]
        book_lo = books_by_threshold[t_lo]
        book_hi = books_by_threshold[t_hi]

        # lower threshold should have higher (or equal) price
        # violation: if we can buy the lower-threshold contract cheaper than selling the higher
        if book_lo.best_ask is not None and book_hi.best_bid is not None:
            if book_lo.best_ask < book_hi.best_bid:
                # buy low-threshold YES (cheap), sell high-threshold YES (expensive)
                edge = book_hi.best_bid - book_lo.best_ask
                size = min(
                    book_lo.ask_volumes[0] if len(book_lo.ask_volumes) > 0 else 0,
                    book_hi.bid_volumes[0] if len(book_hi.bid_volumes) > 0 else 0,
                )
                if size > 0:
                    opps.append(ArbitrageOpportunity(
                        type='monotonicity',
                        pair_id=pair_id,
                        edge=edge,
                        max_size=size,
                        prices=(book_lo.best_ask, book_hi.best_bid),
                    ))

    return opps


def scan_all_opportunities(matched_pairs, books_kalshi, books_poly, min_edge=0.01):
    # top-level scan: iterate all matched pairs, run cross-platform detection
    all_opps = []

    for pair_id, pair in enumerate(matched_pairs):
        if pair_id not in books_kalshi or pair_id not in books_poly:
            continue

        k_data = books_kalshi[pair_id]
        p_data = books_poly[pair_id]

        # build orderbook objects if they're raw dicts
        if isinstance(k_data, dict):
            k_data = OrderBook.from_dict(k_data)
        if isinstance(p_data, dict):
            p_data = OrderBook.from_dict(p_data)

        opps = detect_cross_platform_arb(k_data, p_data, pair_id, min_edge=min_edge)
        all_opps.extend(opps)

    # sort by edge descending, best opportunities first
    all_opps.sort(key=lambda o: o.edge, reverse=True)
    return all_opps


# not production quality, just enough for the paper
if __name__ == '__main__':
    from simulate_markets import simulate_order_book

    rng = np.random.default_rng(42)

    # make two books with a gap (should trigger arb)
    k_book = simulate_order_book(0.40, spread_cents=2, rng=rng)
    p_book = simulate_order_book(0.42, spread_cents=2, rng=rng)

    ob_k = OrderBook.from_dict(k_book)
    ob_p = OrderBook.from_dict(p_book)

    print(f"kalshi: bid={ob_k.best_bid:.2f} ask={ob_k.best_ask:.2f}")
    print(f"poly:   bid={ob_p.best_bid:.2f} ask={ob_p.best_ask:.2f}")
    print(f"spread: kalshi={ob_k.spread:.3f}  poly={ob_p.spread:.3f}")

    # test cost to fill
    avg_px, filled = ob_k.cost_to_fill('ask', 200)
    print(f"\ncost to fill 200 on kalshi ask: avg={avg_px:.4f}, filled={filled}")

    # test arb detection
    opps = detect_cross_platform_arb(ob_k, ob_p, pair_id=0, min_edge=0.001)
    print(f"\ncross-platform opps found: {len(opps)}")
    for o in opps:
        print(f"  edge={o.edge:.4f}  size={o.max_size}  prices={o.prices}")
