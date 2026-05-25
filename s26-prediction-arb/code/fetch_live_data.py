#!/usr/bin/env python3
# fetch_live_data.py -- pull real markets from kalshi + polymarket, run matching pipeline
# read-only, no auth, no trading

import os
import sys
import json
import csv
import time
import requests
from datetime import datetime

# make sure we can import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matching import HybridMatcher, extract_kalshi_fields, extract_polymarket_fields
from fees import net_edge_cross_platform

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

KALSHI_BASE = 'https://api.elections.kalshi.com/trade-api/v2'
POLY_GAMMA_BASE = 'https://gamma-api.polymarket.com'
POLY_CLOB_BASE = 'https://clob.polymarket.com'

REQUEST_TIMEOUT = 10
PAGE_DELAY = 0.5

# -- category filters --

# kalshi sport tickers start with KX + league code
KALSHI_SPORT_PREFIXES = ['KXNFL', 'KXNBA', 'KXMLB', 'KXNHL', 'KXMMA', 'KXUFC']
KALSHI_CRYPTO_KEYWORDS = ['BTC', 'ETH']

# polymarket: just look for keywords in the question text
POLY_SPORT_KEYWORDS = [
    'nfl', 'nba', 'mlb', 'nhl', 'mma', 'ufc',
    'chiefs', 'bills', '49ers', 'eagles', 'cowboys', 'ravens', 'lions', 'dolphins',
    'celtics', 'nuggets', 'bucks', 'suns', 'warriors', 'lakers', 'timberwolves', 'thunder',
    'dodgers', 'braves', 'astros', 'yankees', 'rangers', 'orioles',
    'packers', 'bears', 'steelers', 'bengals', 'jets', 'patriots', 'broncos', 'chargers',
    'nets', 'knicks', 'heat', 'hawks', 'cavaliers', 'mavericks', 'clippers', 'rockets',
    'mets', 'cubs', 'red sox', 'phillies', 'padres', 'guardians',
    'super bowl', 'nba finals', 'world series', 'stanley cup',
]
POLY_CRYPTO_KEYWORDS = ['bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol', 'crypto']


def categorize_kalshi(ticker, title=''):
    """figure out if a kalshi market is sports or crypto (or neither)"""
    ticker_upper = ticker.upper()
    for prefix in KALSHI_SPORT_PREFIXES:
        if ticker_upper.startswith(prefix):
            return 'sports'
    for kw in KALSHI_CRYPTO_KEYWORDS:
        if kw in ticker_upper:
            return 'crypto'
    # newer kalshi tickers use KXMVESPORTS... or KXMVECROSSCATEGORY... prefixes
    if 'SPORT' in ticker_upper or 'GAME' in ticker_upper:
        return 'sports'
    # also check the title for broader coverage
    title_lower = title.lower()
    for kw in ['nfl', 'nba', 'mlb', 'nhl', 'mma', 'ufc',
               'runs scored', 'strikeouts', 'home run', 'touchdown',
               'three-pointer', 'rebound', 'assist']:
        if kw in title_lower:
            return 'sports'
    for kw in ['bitcoin', 'ethereum', 'btc', 'eth', 'crypto']:
        if kw in title_lower:
            return 'crypto'
    return None


def categorize_poly(question):
    """figure out if a polymarket market is sports or crypto"""
    q_lower = question.lower()
    for kw in POLY_SPORT_KEYWORDS:
        if kw in q_lower:
            return 'sports'
    for kw in POLY_CRYPTO_KEYWORDS:
        if kw in q_lower:
            return 'crypto'
    return None


# -- fetching --

def fetch_kalshi_markets(max_markets=1000):
    """paginate through kalshi's public markets endpoint"""
    print(f"\n--- fetching kalshi markets (up to {max_markets}) ---")
    all_markets = []
    cursor = None
    page = 0

    while len(all_markets) < max_markets:
        params = {'limit': 200, 'status': 'open'}
        if cursor:
            params['cursor'] = cursor

        try:
            resp = requests.get(
                f'{KALSHI_BASE}/markets',
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  [error] kalshi page {page}: {e}")
            break

        markets = data.get('markets', [])
        if not markets:
            print(f"  page {page}: no more markets")
            break

        all_markets.extend(markets)
        page += 1
        print(f"  page {page}: got {len(markets)} markets (total: {len(all_markets)})")

        # check for next cursor
        cursor = data.get('cursor')
        if not cursor:
            break

        time.sleep(PAGE_DELAY)

    # filter to sports and crypto
    filtered = []
    for m in all_markets:
        ticker = m.get('ticker', '')
        title = m.get('title', '')
        cat = categorize_kalshi(ticker, title)
        if cat:
            # kalshi prices -- they use _dollars suffix now
            yes_bid = m.get('yes_bid_dollars', m.get('yes_bid'))
            yes_ask = m.get('yes_ask_dollars', m.get('yes_ask'))
            no_bid = m.get('no_bid_dollars', m.get('no_bid'))
            no_ask = m.get('no_ask_dollars', m.get('no_ask'))

            filtered.append({
                'ticker': ticker,
                'title': title,
                'subtitle': m.get('subtitle', ''),
                'event_ticker': m.get('event_ticker', ''),
                'yes_bid': yes_bid,
                'yes_ask': yes_ask,
                'no_bid': no_bid,
                'no_ask': no_ask,
                'close_time': m.get('close_time', ''),
                'volume': m.get('volume', 0),
                'category': cat,
            })

    print(f"  filtered: {len(filtered)} sport/crypto markets out of {len(all_markets)} total")
    return filtered, all_markets


def fetch_polymarket_markets(max_markets=1000):
    """paginate through polymarket gamma api"""
    print(f"\n--- fetching polymarket markets (up to {max_markets}) ---")
    all_markets = []
    offset = 0
    page_size = 100
    page = 0

    while len(all_markets) < max_markets:
        params = {
            'limit': page_size,
            'active': 'true',
            'offset': offset,
        }

        try:
            resp = requests.get(
                f'{POLY_GAMMA_BASE}/markets',
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  [error] polymarket page {page}: {e}")
            break

        # gamma returns a list directly
        if isinstance(data, list):
            markets = data
        elif isinstance(data, dict):
            markets = data.get('markets', data.get('data', []))
        else:
            print(f"  [error] unexpected response type: {type(data)}")
            break

        if not markets:
            print(f"  page {page}: no more markets")
            break

        all_markets.extend(markets)
        page += 1
        offset += page_size
        print(f"  page {page}: got {len(markets)} markets (total: {len(all_markets)})")

        if len(markets) < page_size:
            # last page
            break

        time.sleep(PAGE_DELAY)

    # filter to sports and crypto
    filtered = []
    for m in all_markets:
        question = m.get('question', '')
        cat = categorize_poly(question)
        if cat:
            # parse outcome prices -- can be a json string or already parsed
            outcome_prices = m.get('outcomePrices', '[]')
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, TypeError):
                    outcome_prices = []

            # get clob token ids for order book fetching
            clob_token_ids = m.get('clobTokenIds', '[]')
            if isinstance(clob_token_ids, str):
                try:
                    clob_token_ids = json.loads(clob_token_ids)
                except (json.JSONDecodeError, TypeError):
                    clob_token_ids = []

            filtered.append({
                'question': question,
                'condition_id': m.get('conditionId', ''),
                'slug': m.get('slug', ''),
                'outcome_prices': outcome_prices,
                'clob_token_ids': clob_token_ids,
                'end_date': m.get('endDate', ''),
                'volume': m.get('volume', 0),
                'liquidity': m.get('liquidity', 0),
                'category': cat,
            })

    print(f"  filtered: {len(filtered)} sport/crypto markets out of {len(all_markets)} total")
    return filtered, all_markets


# -- matching --

def run_matching(kalshi_markets, poly_markets):
    """match kalshi markets against polymarket markets using the hybrid pipeline"""
    print(f"\n--- running matching pipeline ---")
    print(f"  kalshi markets: {len(kalshi_markets)}")
    print(f"  polymarket markets: {len(poly_markets)}")

    # skip llm verification for now, keep regex + embedding + fuzzy
    matcher = HybridMatcher(
        threshold=0.3,  # low threshold to catch candidates, we filter later
        use_llm=False,
    )

    results = []
    total_pairs = 0
    start = time.time()

    for i, km in enumerate(kalshi_markets):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  matching kalshi market {i+1}/{len(kalshi_markets)}: {km['ticker'][:40]}...")

        # only match against polymarket markets in the same category
        same_cat = [pm for pm in poly_markets if pm['category'] == km['category']]

        for pm in same_cat:
            total_pairs += 1

            t0 = time.time()
            # the matcher's extract_kalshi_fields won't parse new-format tickers
            # (KXMVESPORTS... hashes). so we try the ticker first but if it doesn't
            # parse, fall back to using the title text which is what we really care about.
            # for the embedding + fuzzy stages, title is better anyway.
            kalshi_text = km['ticker']
            parsed = extract_kalshi_fields(kalshi_text)
            if parsed is None:
                # use title as the "ticker" -- the fuzzy and embedding stages
                # work on raw text so this is fine. regex stage will just score 0.
                kalshi_text = km['title']
            poly_text = pm['question']

            match_result = matcher.match(kalshi_text, poly_text)
            elapsed_ms = (time.time() - t0) * 1000

            if match_result.confidence > 0.3:
                results.append({
                    'kalshi_ticker': km['ticker'],
                    'kalshi_title': km['title'],
                    'kalshi_subtitle': km.get('subtitle', ''),
                    'kalshi_yes_bid': km.get('yes_bid'),
                    'kalshi_yes_ask': km.get('yes_ask'),
                    'kalshi_category': km['category'],
                    'poly_question': pm['question'],
                    'poly_condition_id': pm['condition_id'],
                    'poly_clob_token_ids': pm.get('clob_token_ids', []),
                    'poly_outcome_prices': pm.get('outcome_prices', []),
                    'poly_category': pm['category'],
                    'match_confidence': round(match_result.confidence, 4),
                    'method_used': match_result.method,
                    'time_ms': round(elapsed_ms, 2),
                })

    elapsed_total = time.time() - start
    print(f"  evaluated {total_pairs} pairs in {elapsed_total:.1f}s")
    print(f"  found {len(results)} candidates with confidence > 0.3")

    # sort by confidence descending
    results.sort(key=lambda r: r['match_confidence'], reverse=True)
    return results


# -- order book fetching --

def fetch_kalshi_orderbook(ticker):
    """fetch kalshi order book for a single market"""
    try:
        resp = requests.get(
            f'{KALSHI_BASE}/markets/{ticker}/orderbook',
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"    [error] kalshi orderbook for {ticker}: {e}")
        return None


def fetch_poly_orderbook(token_id):
    """fetch polymarket CLOB order book"""
    try:
        resp = requests.get(
            f'{POLY_CLOB_BASE}/book',
            params={'token_id': token_id},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"    [error] poly orderbook for {token_id[:20]}...: {e}")
        return None


def parse_kalshi_book(data):
    """extract best bid/ask from kalshi orderbook response"""
    if not data:
        return None, None
    ob = data.get('orderbook', data)
    # kalshi returns yes/no arrays
    yes_bids = ob.get('yes', [])
    no_bids = ob.get('no', [])

    best_yes_bid = None
    best_yes_ask = None

    # yes side has bids (someone wants to buy YES)
    if yes_bids:
        # format can vary -- sometimes list of [price, qty], sometimes list of dicts
        if isinstance(yes_bids[0], list):
            prices = [float(b[0]) for b in yes_bids]
        elif isinstance(yes_bids[0], dict):
            prices = [float(b.get('price', 0)) for b in yes_bids]
        else:
            prices = [float(b) for b in yes_bids]
        if prices:
            best_yes_bid = max(prices)

    # no bids = yes asks effectively (1 - no_bid = yes_ask)
    if no_bids:
        if isinstance(no_bids[0], list):
            prices = [float(b[0]) for b in no_bids]
        elif isinstance(no_bids[0], dict):
            prices = [float(b.get('price', 0)) for b in no_bids]
        else:
            prices = [float(b) for b in no_bids]
        if prices:
            best_yes_ask = 1.0 - min(prices) if min(prices) < 1.0 else None

    return best_yes_bid, best_yes_ask


def parse_poly_book(data):
    """extract best bid/ask from polymarket CLOB response"""
    if not data:
        return None, None

    bids = data.get('bids', [])
    asks = data.get('asks', [])

    best_bid = None
    best_ask = None

    if bids:
        if isinstance(bids[0], dict):
            prices = [float(b.get('price', 0)) for b in bids]
        else:
            prices = [float(b) for b in bids]
        if prices:
            best_bid = max(prices)

    if asks:
        if isinstance(asks[0], dict):
            prices = [float(a.get('price', 0)) for a in asks]
        else:
            prices = [float(a) for a in asks]
        if prices:
            best_ask = min(prices)

    return best_bid, best_ask


def fetch_orderbooks_for_top_matches(matches, n=5):
    """pull order books for the top n matches and compute edges"""
    print(f"\n--- fetching order books for top {n} matches ---")

    results = []
    fetched = 0

    for match in matches:
        if fetched >= n:
            break

        ticker = match['kalshi_ticker']
        clob_ids = match.get('poly_clob_token_ids', [])

        if not clob_ids:
            print(f"  skipping {ticker}: no clob token ids for polymarket side")
            continue

        # use first token id (YES outcome)
        token_id = clob_ids[0] if clob_ids else None
        if not token_id:
            continue

        print(f"  [{fetched+1}] {ticker} <-> {match['poly_question'][:60]}...")

        # fetch both books
        k_book_raw = fetch_kalshi_orderbook(ticker)
        time.sleep(PAGE_DELAY)
        p_book_raw = fetch_poly_orderbook(token_id)
        time.sleep(PAGE_DELAY)

        k_bid, k_ask = parse_kalshi_book(k_book_raw)
        p_bid, p_ask = parse_poly_book(p_book_raw)

        # fall back to prices from market data if orderbook didn't return anything
        # kalshi prices are already in dollars (0.0000 to 1.0000 range)
        if k_bid is None and match.get('kalshi_yes_bid') is not None:
            try:
                val = float(match['kalshi_yes_bid'])
                k_bid = val if val <= 1.0 else val / 100.0
            except (ValueError, TypeError):
                pass
        if k_ask is None and match.get('kalshi_yes_ask') is not None:
            try:
                val = float(match['kalshi_yes_ask'])
                k_ask = val if val <= 1.0 else val / 100.0
            except (ValueError, TypeError):
                pass

        # compute gross edge if we have both sides
        gross_edge = None
        net_edge = None

        # direction 1: buy YES kalshi, sell YES poly (buy NO poly)
        if k_ask is not None and p_bid is not None:
            gross_1 = 1.0 - k_ask - (1.0 - p_bid)  # = p_bid - k_ask
            net_1 = net_edge_cross_platform(k_ask, 1.0 - p_bid, 100)
        else:
            gross_1 = None
            net_1 = None

        # direction 2: buy YES poly, sell YES kalshi (buy NO kalshi)
        if p_ask is not None and k_bid is not None:
            gross_2 = 1.0 - p_ask - (1.0 - k_bid)  # = k_bid - p_ask
            net_2 = net_edge_cross_platform(p_ask, 1.0 - k_bid, 100)
        else:
            gross_2 = None
            net_2 = None

        # pick the better direction
        candidates = []
        if gross_1 is not None:
            candidates.append(('buy_kalshi_yes', gross_1, net_1))
        if gross_2 is not None:
            candidates.append(('buy_poly_yes', gross_2, net_2))

        if candidates:
            best = max(candidates, key=lambda x: x[1] if x[1] is not None else -999)
            direction, gross_edge, net_edge = best
        else:
            direction = 'n/a'

        result = {
            'kalshi_ticker': ticker,
            'poly_question': match['poly_question'][:80],
            'match_confidence': match['match_confidence'],
            'kalshi_best_bid': k_bid,
            'kalshi_best_ask': k_ask,
            'poly_best_bid': p_bid,
            'poly_best_ask': p_ask,
            'best_direction': direction,
            'gross_edge': round(gross_edge, 4) if gross_edge is not None else None,
            'net_edge': round(net_edge, 4) if net_edge is not None else None,
        }
        results.append(result)
        fetched += 1

    return results


# -- output --

def save_results(kalshi_raw, poly_raw, kalshi_filtered, poly_filtered, matches, orderbook_results):
    """save everything to data/ directory"""
    print(f"\n--- saving results ---")

    # raw market data
    kalshi_path = os.path.join(DATA_DIR, 'kalshi_markets.json')
    with open(kalshi_path, 'w') as f:
        json.dump(kalshi_filtered, f, indent=2, default=str)
    print(f"  saved {len(kalshi_filtered)} kalshi markets to {kalshi_path}")

    poly_path = os.path.join(DATA_DIR, 'polymarket_markets.json')
    with open(poly_path, 'w') as f:
        json.dump(poly_filtered, f, indent=2, default=str)
    print(f"  saved {len(poly_filtered)} polymarket markets to {poly_path}")

    # match candidates (confidence > 0.5)
    strong_matches = [m for m in matches if m['match_confidence'] > 0.5]
    csv_path = os.path.join(DATA_DIR, 'match_candidates.csv')
    if strong_matches:
        keys = ['kalshi_ticker', 'kalshi_title', 'poly_question', 'match_confidence',
                'method_used', 'time_ms', 'kalshi_category', 'poly_category']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(strong_matches)
        print(f"  saved {len(strong_matches)} match candidates (conf > 0.5) to {csv_path}")
    else:
        print(f"  no matches with confidence > 0.5 to save")

    # all matches for reference
    all_matches_path = os.path.join(DATA_DIR, 'all_match_results.json')
    with open(all_matches_path, 'w') as f:
        json.dump(matches, f, indent=2, default=str)
    print(f"  saved {len(matches)} total match results to {all_matches_path}")

    # orderbook results
    if orderbook_results:
        ob_path = os.path.join(DATA_DIR, 'orderbook_analysis.json')
        with open(ob_path, 'w') as f:
            json.dump(orderbook_results, f, indent=2, default=str)
        print(f"  saved {len(orderbook_results)} orderbook analyses to {ob_path}")


def print_summary(kalshi_filtered, poly_filtered, matches, orderbook_results):
    """print a human-readable summary"""
    print(f"\n{'='*70}")
    print(f"  LIVE DATA FETCH SUMMARY")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    print(f"\nmarkets fetched:")
    k_sports = sum(1 for m in kalshi_filtered if m['category'] == 'sports')
    k_crypto = sum(1 for m in kalshi_filtered if m['category'] == 'crypto')
    p_sports = sum(1 for m in poly_filtered if m['category'] == 'sports')
    p_crypto = sum(1 for m in poly_filtered if m['category'] == 'crypto')
    print(f"  kalshi:      {len(kalshi_filtered):>4} total ({k_sports} sports, {k_crypto} crypto)")
    print(f"  polymarket:  {len(poly_filtered):>4} total ({p_sports} sports, {p_crypto} crypto)")

    if not matches:
        print(f"\nno match candidates found")
        return

    # confidence distribution
    confs = [m['match_confidence'] for m in matches]
    strong = [c for c in confs if c > 0.5]
    medium = [c for c in confs if 0.3 < c <= 0.5]

    print(f"\nmatch candidates:")
    print(f"  total evaluated pairs: (kalshi x poly within same category)")
    print(f"  candidates (conf > 0.3): {len(matches)}")
    print(f"  strong (conf > 0.5):     {len(strong)}")
    print(f"  medium (0.3 < conf <= 0.5): {len(medium)}")

    if confs:
        print(f"\nconfidence distribution:")
        # histogram buckets
        buckets = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        for lo, hi in buckets:
            count = sum(1 for c in confs if lo <= c < hi)
            bar = '#' * count
            print(f"  [{lo:.1f}-{hi:.1f}): {count:>4} {bar}")

    # top 20 matches
    print(f"\ntop 20 highest-confidence matches:")
    print(f"  {'conf':>5}  {'kalshi ticker':<35} {'polymarket question':<50}")
    print(f"  {'-'*5}  {'-'*35} {'-'*50}")
    for m in matches[:20]:
        conf = m['match_confidence']
        ticker = m['kalshi_ticker'][:35]
        question = m['poly_question'][:50]
        print(f"  {conf:.3f}  {ticker:<35} {question}")

    # orderbook analysis
    if orderbook_results:
        print(f"\norder book analysis (top matched pairs):")
        print(f"  {'ticker':<25} {'k_bid':>6} {'k_ask':>6} {'p_bid':>6} {'p_ask':>6} {'gross':>7} {'net':>7}")
        print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")
        for ob in orderbook_results:
            ticker = ob['kalshi_ticker'][:25]
            k_bid = f"{ob['kalshi_best_bid']:.3f}" if ob['kalshi_best_bid'] is not None else '  n/a'
            k_ask = f"{ob['kalshi_best_ask']:.3f}" if ob['kalshi_best_ask'] is not None else '  n/a'
            p_bid = f"{ob['poly_best_bid']:.3f}" if ob['poly_best_bid'] is not None else '  n/a'
            p_ask = f"{ob['poly_best_ask']:.3f}" if ob['poly_best_ask'] is not None else '  n/a'
            gross = f"{ob['gross_edge']:.4f}" if ob['gross_edge'] is not None else '   n/a'
            net = f"{ob['net_edge']:.4f}" if ob['net_edge'] is not None else '   n/a'
            print(f"  {ticker:<25} {k_bid:>6} {k_ask:>6} {p_bid:>6} {p_ask:>6} {gross:>7} {net:>7}")
            print(f"    -> {ob['poly_question']}")
            print(f"    -> direction: {ob['best_direction']}")


def main():
    print("fetch_live_data.py -- pulling real market data")
    print(f"timestamp: {datetime.now().isoformat()}")
    t_start = time.time()

    # 1. fetch markets
    kalshi_filtered, kalshi_raw = fetch_kalshi_markets(max_markets=1000)
    poly_filtered, poly_raw = fetch_polymarket_markets(max_markets=1000)

    if not kalshi_filtered and not poly_filtered:
        print("\n[!] no markets fetched from either platform. check network / api status.")
        return

    # 2. run matching pipeline
    matches = []
    if kalshi_filtered and poly_filtered:
        matches = run_matching(kalshi_filtered, poly_filtered)
    else:
        print("\n[!] need markets from both platforms to run matching")

    # 3. fetch order books for top matches
    orderbook_results = []
    strong_matches = [m for m in matches if m['match_confidence'] > 0.5]
    if strong_matches:
        orderbook_results = fetch_orderbooks_for_top_matches(strong_matches, n=5)
    else:
        # try top matches even if below 0.5
        if matches:
            print("\n  no matches above 0.5, trying top matches anyway...")
            orderbook_results = fetch_orderbooks_for_top_matches(matches[:5], n=5)

    # 4. save and summarize
    save_results(kalshi_raw, poly_raw, kalshi_filtered, poly_filtered, matches, orderbook_results)
    print_summary(kalshi_filtered, poly_filtered, matches, orderbook_results)

    elapsed = time.time() - t_start
    print(f"\ntotal runtime: {elapsed:.1f}s")
    print("done.")


if __name__ == '__main__':
    main()
