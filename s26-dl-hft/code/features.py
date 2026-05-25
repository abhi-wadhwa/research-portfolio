import numpy as np


def compute_ofi(bid_volumes, ask_volumes, n_levels=5):
    # OFI at multiple levels. positive = buy pressure
    delta_bid = np.diff(bid_volumes[:, :n_levels], axis=0)
    delta_ask = np.diff(ask_volumes[:, :n_levels], axis=0)
    ofi = delta_bid - delta_ask
    ofi = np.vstack([np.zeros((1, n_levels)), ofi])
    return ofi


def compute_microprice(bid_volumes, ask_volumes, mid_prices, tick_size=0.01):
    # volume-weighted mid, better than simple mid
    best_bid_vol = bid_volumes[:, 0]
    best_ask_vol = ask_volumes[:, 0]
    spread = tick_size

    best_bid = mid_prices - spread / 2
    best_ask = mid_prices + spread / 2

    total_vol = best_bid_vol + best_ask_vol
    total_vol = np.maximum(total_vol, 1e-10)

    microprice = (best_ask_vol * best_bid + best_bid_vol * best_ask) / total_vol
    return microprice


def compute_vpin(mid_prices, volumes, n_buckets=50, bucket_size=None):
    # VPIN, see easley et al 2012
    T = len(mid_prices)
    if bucket_size is None:
        bucket_size = max(volumes.sum() / (4 * n_buckets), 1.0)

    # tick rule for trade classification
    price_changes = np.diff(mid_prices, prepend=mid_prices[0])
    buy_indicator = (price_changes > 0).astype(float)
    sell_indicator = (price_changes < 0).astype(float)
    tie_mask = price_changes == 0
    buy_indicator[tie_mask] = 0.5
    sell_indicator[tie_mask] = 0.5

    buy_vol = buy_indicator * volumes
    sell_vol = sell_indicator * volumes

    cum_vol = np.cumsum(volumes)
    bucket_ids = (cum_vol / bucket_size).astype(int)

    max_bucket = bucket_ids[-1] + 1
    bucket_buy = np.zeros(max_bucket)
    bucket_sell = np.zeros(max_bucket)

    # TODO: vectorize this, it's slow for large T
    for i in range(T):
        b = min(bucket_ids[i], max_bucket - 1)
        bucket_buy[b] += buy_vol[i]
        bucket_sell[b] += sell_vol[i]

    # rolling VPIN
    vpin_buckets = np.zeros(max_bucket)
    for b in range(max_bucket):
        start = max(0, b - n_buckets + 1)
        total_buy = bucket_buy[start:b + 1].sum()
        total_sell = bucket_sell[start:b + 1].sum()
        total = total_buy + total_sell
        if total > 0:
            vpin_buckets[b] = abs(total_buy - total_sell) / total

    vpin = np.zeros(T)
    for i in range(T):
        b = min(bucket_ids[i], max_bucket - 1)
        vpin[i] = vpin_buckets[b]

    return vpin


def compute_hawkes_intensity(events, timestamps, mu=1.0, alpha=0.5, beta=10.0):
    intensity = np.full(len(timestamps), mu)
    event_arr = np.array(events)

    if len(event_arr) == 0:
        return intensity

    for i, t in enumerate(timestamps):
        past = event_arr[event_arr < t]
        if len(past) > 0:
            # only recent events, old ones are negligible
            recent = past[past > t - 5.0 / beta]
            if len(recent) > 0:
                intensity[i] = mu + alpha * beta * np.sum(
                    np.exp(-beta * (t - recent))
                )

    return intensity


def compute_spread_regime(bid_volumes, ask_volumes, mid_prices):
    # tight(0), normal(1), wide(2) based on vol imbalance + price vol
    T = len(mid_prices)
    window = min(50, T // 10)

    best_vol = bid_volumes[:, 0] + ask_volumes[:, 0]

    vol = np.zeros(T)
    for i in range(window, T):
        vol[i] = np.std(mid_prices[i - window:i])

    vol_norm = (best_vol - np.mean(best_vol)) / (np.std(best_vol) + 1e-10)
    price_vol_norm = (vol - np.mean(vol)) / (np.std(vol) + 1e-10)

    score = vol_norm - price_vol_norm

    # tertile split
    regime = np.ones(T, dtype=int)
    p33 = np.percentile(score, 33)
    p66 = np.percentile(score, 66)
    regime[score > p66] = 0
    regime[score < p33] = 2

    return regime


# not sure if threshold should be relative or absolute here
def compute_midprice_labels(mid_prices, horizon=10, threshold=0.0001):
    # 0=down, 1=flat, 2=up
    T = len(mid_prices)
    labels = np.ones(T, dtype=int)

    for i in range(T - horizon):
        ret = (mid_prices[i + horizon] - mid_prices[i]) / (mid_prices[i] + 1e-10)
        if ret > threshold:
            labels[i] = 2
        elif ret < -threshold:
            labels[i] = 0

    return labels


def extract_all_features(data, horizon=10):
    bid_vol = data['bid_volumes']
    ask_vol = data['ask_volumes']
    mid = data['mid_prices']
    n_levels = data.get('n_levels', bid_vol.shape[1])

    ofi = compute_ofi(bid_vol, ask_vol, n_levels=n_levels)
    microprice = compute_microprice(bid_vol, ask_vol, mid)

    # proxy trade volume
    total_vol = bid_vol.sum(axis=1) + ask_vol.sum(axis=1)
    trade_vol = np.abs(np.diff(total_vol, prepend=total_vol[0]))
    trade_vol = np.maximum(trade_vol, 0.1)

    vpin = compute_vpin(mid, trade_vol)

    if 'events' in data:
        buy_mo = data['events'][4] if len(data['events']) > 4 else []
        sell_mo = data['events'][5] if len(data['events']) > 5 else []
        all_trades = sorted(buy_mo + sell_mo)
        intensity = compute_hawkes_intensity(all_trades, data['timestamps'])
    else:
        intensity = np.ones(len(mid))

    midprice_labels = compute_midprice_labels(mid, horizon=horizon)
    spread_labels = compute_spread_regime(bid_vol, ask_vol, mid)

    # features: OFI(n_levels) + microprice + vpin + intensity + bid_vol + ask_vol
    feature_matrix = np.column_stack([
        ofi,
        microprice[:, None],
        vpin[:, None],
        intensity[:, None],
        bid_vol,
        ask_vol
    ])

    return {
        'features': feature_matrix,
        'ofi': ofi,
        'microprice': microprice,
        'vpin': vpin,
        'intensity': intensity,
        'midprice_labels': midprice_labels,
        'spread_labels': spread_labels,
        'mid_prices': mid,
        'timestamps': data['timestamps'],
        'bid_volumes': bid_vol,
        'ask_volumes': ask_vol,
        'n_features': feature_matrix.shape[1],
        'n_levels': n_levels
    }


if __name__ == '__main__':
    from simulate_lob import generate_lob_data

    data = generate_lob_data(T=5.0, seed=42)
    feats = extract_all_features(data, horizon=10)

    print(f"Feature matrix shape: {feats['features'].shape}")
    print(f"Number of features: {feats['n_features']}")
    print(f"OFI shape: {feats['ofi'].shape}")
    print(f"VPIN range: [{feats['vpin'].min():.4f}, {feats['vpin'].max():.4f}]")
    print(f"Intensity range: [{feats['intensity'].min():.2f}, {feats['intensity'].max():.2f}]")
    print(f"Mid-price labels distribution: {np.bincount(feats['midprice_labels'])}")
    print(f"Spread regime distribution: {np.bincount(feats['spread_labels'])}")
