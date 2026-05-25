import math
import numpy as np


# kalshi charges per-contract taker fee, capped at 7% of expected variance
# see https://kalshi.com/docs/fees for the formula
def kalshi_fee(contracts, price):
    # ceil(0.07 * C * P * (1-P)) in cents
    raw = 0.07 * contracts * price * (1 - price)
    return math.ceil(raw)


def kalshi_fee_dollars(contracts, price):
    # just the cents version / 100
    return kalshi_fee(contracts, price) / 100.0


# polymarket uses a simpler proportional fee
# coefficient defaults to 2% but they change it sometimes
def polymarket_fee(quantity, price, coefficient=0.02):
    return coefficient * quantity * price * (1 - price)


def net_edge_cross_platform(a_yes, b_no, size, poly_coeff=0.02):
    # buying YES on kalshi at a_yes, NO on polymarket at b_no
    # gross edge: if both resolve correctly we get $1 and paid a_yes + b_no
    gross = 1.0 - a_yes - b_no

    # subtract fees per contract
    k_fee = kalshi_fee_dollars(size, a_yes) / size
    p_fee = polymarket_fee(1, b_no, poly_coeff)  # per-unit already

    return gross - k_fee - p_fee


def net_edge_same_platform(p_yes, p_no, size):
    # kalshi implied arb: buy both YES and NO when they sum to < $1
    gross = 1.0 - p_yes - p_no

    # pay fees on both legs
    k_fee_yes = kalshi_fee_dollars(size, p_yes) / size
    k_fee_no = kalshi_fee_dollars(size, p_no) / size

    return gross - k_fee_yes - k_fee_no


def fee_impact_curve(prices, contracts=100):
    # generate fee arrays over a price range, used for the fee comparison figure
    prices = np.asarray(prices)
    kalshi_fees = np.array([kalshi_fee_dollars(contracts, p) for p in prices])
    poly_fees = np.array([polymarket_fee(contracts, p) for p in prices])

    return {
        'kalshi_fees': kalshi_fees,
        'poly_fees': poly_fees,
    }


# quick sanity check
if __name__ == '__main__':
    # should be symmetric around 0.5
    for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
        kf = kalshi_fee(100, p)
        pf = polymarket_fee(100, p)
        print(f"p={p:.2f}  kalshi={kf}c  poly=${pf:.4f}")

    # check a realistic arb
    edge = net_edge_cross_platform(0.45, 0.52, 100)
    print(f"\ncross-platform edge: ${edge:.4f}/contract")

    edge2 = net_edge_same_platform(0.45, 0.52, 100)
    print(f"same-platform edge:  ${edge2:.4f}/contract")
