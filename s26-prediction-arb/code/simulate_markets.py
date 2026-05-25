import numpy as np
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


# ---- hardcoded team/entity data ----
# not exhaustive, just enough for realistic synthetic data

NFL_TEAMS = [
    ("Kansas City Chiefs", "KC"),
    ("Buffalo Bills", "BUF"),
    ("San Francisco 49ers", "SF"),
    ("Philadelphia Eagles", "PHI"),
    ("Dallas Cowboys", "DAL"),
    ("Baltimore Ravens", "BAL"),
    ("Detroit Lions", "DET"),
    ("Miami Dolphins", "MIA"),
]

NBA_TEAMS = [
    ("Boston Celtics", "BOS"),
    ("Denver Nuggets", "DEN"),
    ("Milwaukee Bucks", "MIL"),
    ("Phoenix Suns", "PHX"),
    ("Golden State Warriors", "GSW"),
    ("Los Angeles Lakers", "LAL"),
    ("Minnesota Timberwolves", "MIN"),
    ("Oklahoma City Thunder", "OKC"),
]

MLB_TEAMS = [
    ("Los Angeles Dodgers", "LAD"),
    ("Atlanta Braves", "ATL"),
    ("Houston Astros", "HOU"),
    ("New York Yankees", "NYY"),
    ("Texas Rangers", "TEX"),
    ("Baltimore Orioles", "BAL"),
]

CRYPTO_ASSETS = [
    ("Bitcoin", "BTC", 40000, 80000),
    ("Ethereum", "ETH", 2000, 5000),
    ("Solana", "SOL", 50, 250),
]

POLITICIANS = [
    ("Kamala Harris", "HARRIS"),
    ("Donald Trump", "TRUMP"),
    ("Ron DeSantis", "DESANTIS"),
    ("Gavin Newsom", "NEWSOM"),
    ("Nikki Haley", "HALEY"),
]

ELECTION_TYPES = ["presidential", "senate", "governor"]

ECON_EVENTS = [
    ("Fed rate", "FED", "rate", 3.0, 6.0, 0.25),
    ("unemployment", "UNEMP", "unemployment rate", 3.0, 6.0, 0.5),
    ("GDP growth", "GDP", "GDP growth", -1.0, 5.0, 0.5),
]

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _random_date(rng):
    # pick a plausible 2025 date
    month_idx = rng.integers(0, 12)
    day = rng.integers(1, 29)  # lazy, avoid month-length issues
    return month_idx, day


def _format_kalshi_date(month_idx, day):
    return f"25{MONTHS[month_idx]}{day:02d}"


def _format_human_date(month_idx, day):
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    return f"{month_names[month_idx]} {day}, 2025"


# ---- pair generators ----

def generate_sports_pair(sport, rng):
    if sport == 'nfl':
        teams = NFL_TEAMS
        prefix = "KXNFLGAME"
    elif sport == 'nba':
        teams = NBA_TEAMS
        prefix = "KXNBAGAME"
    elif sport == 'mlb':
        teams = MLB_TEAMS
        prefix = "KXMLBGAME"
    else:
        raise ValueError(f"unknown sport: {sport}")

    # pick two different teams
    idx = rng.choice(len(teams), size=2, replace=False)
    team_a, abbr_a = teams[idx[0]]
    team_b, abbr_b = teams[idx[1]]

    month_idx, day = _random_date(rng)
    date_str = _format_kalshi_date(month_idx, day)
    human_date = _format_human_date(month_idx, day)

    kalshi_ticker = f"{prefix}-{date_str}-{abbr_a}{abbr_b}"
    poly_desc = f"Will the {team_a} beat the {team_b} on {human_date}?"

    metadata = {
        'category': sport,
        'team_a': team_a,
        'team_b': team_b,
        'abbr_a': abbr_a,
        'abbr_b': abbr_b,
        'date': human_date,
        'month_idx': int(month_idx),
        'day': int(day),
    }
    return kalshi_ticker, poly_desc, metadata


def generate_crypto_pair(rng):
    asset_name, symbol, lo, hi = CRYPTO_ASSETS[rng.integers(0, len(CRYPTO_ASSETS))]
    threshold = int(rng.integers(lo // 1000, hi // 1000 + 1) * 1000)
    month_idx, day = _random_date(rng)

    date_str = _format_kalshi_date(month_idx, day)
    human_date = _format_human_date(month_idx, day)

    kalshi_ticker = f"KX{symbol}-{date_str}-T{threshold}"
    poly_desc = f"Will {asset_name} be above ${threshold:,} on {human_date}?"

    metadata = {
        'category': 'crypto',
        'asset': asset_name,
        'symbol': symbol,
        'threshold': threshold,
        'date': human_date,
        'month_idx': int(month_idx),
        'day': int(day),
    }
    return kalshi_ticker, poly_desc, metadata


def generate_politics_pair(rng):
    candidate_name, candidate_code = POLITICIANS[rng.integers(0, len(POLITICIANS))]
    election = ELECTION_TYPES[rng.integers(0, len(ELECTION_TYPES))]

    kalshi_ticker = f"KXPRES-25-{candidate_code}"
    poly_desc = f"Will {candidate_name} win the 2025 {election} election?"

    metadata = {
        'category': 'politics',
        'candidate': candidate_name,
        'candidate_code': candidate_code,
        'election_type': election,
    }
    return kalshi_ticker, poly_desc, metadata


def generate_econ_pair(rng):
    event_name, code, desc_name, lo, hi, step = ECON_EVENTS[rng.integers(0, len(ECON_EVENTS))]
    n_steps = int((hi - lo) / step) + 1
    threshold = lo + rng.integers(0, n_steps) * step
    threshold = round(threshold, 2)  # float weirdness

    # pick a meeting month (fed meets ~8 times/year, whatever)
    month_idx = rng.integers(0, 12)
    month_name_full = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"][month_idx]

    # format threshold for ticker - remove decimal point
    thresh_str = str(threshold).replace('.', '')
    kalshi_ticker = f"KX{code}-25{MONTHS[month_idx]}-T{thresh_str}"
    poly_desc = f"Will the {desc_name} be above {threshold}% at the {month_name_full} 2025 meeting?"

    metadata = {
        'category': 'economics',
        'event': event_name,
        'code': code,
        'threshold': threshold,
        'month': month_name_full,
        'month_idx': int(month_idx),
    }
    return kalshi_ticker, poly_desc, metadata


# ---- negative pair corruption ----

def generate_negative_pair(pair, corruption, rng):
    # takes a true (ticker, desc, meta) and breaks it so it shouldn't match
    ticker, desc, meta = pair
    cat = meta['category']

    if corruption == 'swap_teams' and cat in ('nfl', 'nba', 'mlb'):
        # swap the team order in the description but not the ticker
        new_desc = desc.replace(meta['team_a'], "PLACEHOLDER")
        new_desc = new_desc.replace(meta['team_b'], meta['team_a'])
        new_desc = new_desc.replace("PLACEHOLDER", meta['team_b'])
        new_meta = {**meta, 'corruption': 'swap_teams'}
        return ticker, new_desc, new_meta

    elif corruption == 'wrong_date':
        # shift the date by a few days in the description
        offset = rng.integers(2, 10)
        new_day = (meta.get('day', 15) + offset) % 28 + 1
        old_date = meta.get('date', '')
        if old_date:
            # hacky string replace for the day number
            parts = old_date.split()
            if len(parts) == 3:
                parts[1] = f"{new_day},"
                new_date = ' '.join(parts)
                new_desc = desc.replace(old_date, new_date)
                new_meta = {**meta, 'corruption': 'wrong_date', 'corrupted_date': new_date}
                return ticker, new_desc, new_meta
        # fallback: just append junk
        new_meta = {**meta, 'corruption': 'wrong_date'}
        return ticker, desc + " (updated)", new_meta

    elif corruption == 'wrong_threshold' and cat in ('crypto', 'economics'):
        # change the threshold in the description
        old_thresh = meta['threshold']
        if cat == 'crypto':
            new_thresh = old_thresh + rng.choice([-5000, -2000, 2000, 5000])
            new_desc = desc.replace(f"${old_thresh:,}", f"${new_thresh:,}")
        else:
            new_thresh = old_thresh + rng.choice([-0.5, -0.25, 0.25, 0.5])
            new_desc = desc.replace(f"{old_thresh}%", f"{new_thresh}%")
        new_meta = {**meta, 'corruption': 'wrong_threshold', 'corrupted_threshold': new_thresh}
        return ticker, new_desc, new_meta

    elif corruption == 'wrong_category':
        # pair a ticker from one category with description from another
        # just generate a fresh pair from a different category
        other_cats = [c for c in ['nfl', 'crypto', 'politics', 'economics'] if c != cat]
        new_cat = rng.choice(other_cats)
        if new_cat in ('nfl', 'nba', 'mlb'):
            _, new_desc, _ = generate_sports_pair(new_cat, rng)
        elif new_cat == 'crypto':
            _, new_desc, _ = generate_crypto_pair(rng)
        elif new_cat == 'politics':
            _, new_desc, _ = generate_politics_pair(rng)
        else:
            _, new_desc, _ = generate_econ_pair(rng)
        new_meta = {**meta, 'corruption': 'wrong_category', 'wrong_cat': new_cat}
        return ticker, new_desc, new_meta

    # if corruption type doesn't apply, just do wrong_category as fallback
    return generate_negative_pair(pair, 'wrong_category', rng)


# ---- benchmark dataset ----

def generate_matching_benchmark(n_pairs=5000, neg_ratio=1.0, seed=42):
    rng = np.random.default_rng(seed)

    # category mix: ~50% sports, 20% crypto, 15% politics, 15% econ
    n_sports = int(n_pairs * 0.5)
    n_crypto = int(n_pairs * 0.2)
    n_politics = int(n_pairs * 0.15)
    n_econ = n_pairs - n_sports - n_crypto - n_politics

    pairs = []  # (ticker, desc, meta)

    # sports split roughly evenly across leagues
    sport_types = ['nfl', 'nba', 'mlb']
    for i in range(n_sports):
        sport = sport_types[i % len(sport_types)]
        pairs.append(generate_sports_pair(sport, rng))

    for _ in range(n_crypto):
        pairs.append(generate_crypto_pair(rng))

    for _ in range(n_politics):
        pairs.append(generate_politics_pair(rng))

    for _ in range(n_econ):
        pairs.append(generate_econ_pair(rng))

    # build positive examples
    records = []
    for ticker, desc, meta in pairs:
        records.append({
            'kalshi_ticker': ticker,
            'poly_desc': desc,
            'is_match': True,
            'category': meta['category'],
            'corruption_type': None,
        })

    # build negative examples
    n_neg = int(n_pairs * neg_ratio)
    corruptions = ['swap_teams', 'wrong_date', 'wrong_threshold', 'wrong_category']
    for i in range(n_neg):
        # pick a random positive pair to corrupt
        src_idx = rng.integers(0, len(pairs))
        corruption = corruptions[rng.integers(0, len(corruptions))]
        neg_ticker, neg_desc, neg_meta = generate_negative_pair(pairs[src_idx], corruption, rng)
        records.append({
            'kalshi_ticker': neg_ticker,
            'poly_desc': neg_desc,
            'is_match': False,
            'category': neg_meta['category'],
            'corruption_type': neg_meta.get('corruption', corruption),
        })

    # shuffle everything
    rng.shuffle(records)
    return records


# ---- order book simulation ----

def simulate_order_book(fair_price, depth=10, spread_cents=3, vol_mean=50, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # half-spread in price units (contracts trade 0-1)
    half_spread = (spread_cents / 2) / 100.0

    best_bid = fair_price - half_spread
    best_ask = fair_price + half_spread

    # generate price levels stepping away from mid
    tick = 0.01  # 1 cent ticks
    bid_prices = np.array([best_bid - i * tick for i in range(depth)])
    ask_prices = np.array([best_ask + i * tick for i in range(depth)])

    # clip to valid range
    bid_prices = np.clip(bid_prices, 0.01, 0.99)
    ask_prices = np.clip(ask_prices, 0.01, 0.99)

    # volumes from lognormal, deeper levels tend to have more size
    bid_volumes = rng.lognormal(np.log(vol_mean), 0.7, size=depth).astype(int)
    ask_volumes = rng.lognormal(np.log(vol_mean), 0.7, size=depth).astype(int)

    # at least 1 contract per level
    bid_volumes = np.maximum(bid_volumes, 1)
    ask_volumes = np.maximum(ask_volumes, 1)

    return {
        'bid_prices': bid_prices,
        'bid_volumes': bid_volumes,
        'ask_prices': ask_prices,
        'ask_volumes': ask_volumes,
    }


# ---- full backtest data ----

def simulate_backtest_data(n_days=90, n_pairs=100, seed=42):
    rng = np.random.default_rng(seed)

    # generate matched pairs (just use a small benchmark)
    benchmark = generate_matching_benchmark(n_pairs=n_pairs, neg_ratio=0, seed=seed)
    matched_pairs = [r for r in benchmark if r['is_match']]

    # initial fair prices
    fair_prices = rng.uniform(0.15, 0.85, size=n_pairs)

    daily_books = {}
    price_history = np.zeros((n_days, n_pairs))

    for day in range(n_days):
        # random walk step, bounded
        step = rng.normal(0, 0.02, size=n_pairs)
        fair_prices = fair_prices + step
        fair_prices = np.clip(fair_prices, 0.01, 0.99)
        price_history[day] = fair_prices

        day_books = {}
        for pair_idx in range(n_pairs):
            # each pair gets a kalshi book and a poly book
            # poly tracks kalshi but with some noise (cross-platform spread)
            poly_offset = rng.normal(0, 0.01)
            poly_fair = np.clip(fair_prices[pair_idx] + poly_offset, 0.01, 0.99)

            k_book = simulate_order_book(fair_prices[pair_idx], rng=rng)
            p_book = simulate_order_book(poly_fair, rng=rng)

            day_books[pair_idx] = {
                'kalshi': k_book,
                'polymarket': p_book,
            }

        daily_books[day] = day_books

    # simulate resolutions - events resolve on their "date" or at end of period
    # for simplicity: 30% resolve during the 90-day window
    resolutions = {}
    for pair_idx in range(n_pairs):
        if rng.random() < 0.3:
            resolve_day = rng.integers(n_days // 3, n_days)
            # resolve YES if final price was > 0.5, NO otherwise
            resolved_yes = price_history[resolve_day, pair_idx] > 0.5
            resolutions[pair_idx] = {
                'day': int(resolve_day),
                'outcome': 'YES' if resolved_yes else 'NO',
                'final_price': float(price_history[resolve_day, pair_idx]),
            }

    return {
        'matched_pairs': matched_pairs,
        'daily_books': daily_books,
        'resolutions': resolutions,
        'price_history': price_history,
    }


# works for the paper
if __name__ == '__main__':
    print("generating matching benchmark...")
    bench = generate_matching_benchmark(n_pairs=100, seed=123)
    pos = sum(1 for r in bench if r['is_match'])
    neg = sum(1 for r in bench if not r['is_match'])
    print(f"  {pos} positive, {neg} negative pairs")
    print(f"  sample: {bench[0]}")

    print("\nsimulating order book...")
    book = simulate_order_book(0.55, rng=np.random.default_rng(0))
    print(f"  bids: {book['bid_prices'][:3]}  asks: {book['ask_prices'][:3]}")

    print("\nsimulating backtest data (small)...")
    bt = simulate_backtest_data(n_days=10, n_pairs=5, seed=0)
    print(f"  {len(bt['matched_pairs'])} pairs, {len(bt['daily_books'])} days")
    print(f"  {len(bt['resolutions'])} resolved events")
