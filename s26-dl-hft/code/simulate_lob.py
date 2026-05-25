import numpy as np
from scipy.optimize import minimize


class HawkesProcess:
    def __init__(self, n_dims, mu, alpha, beta):
        self.n_dims = n_dims
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

    def simulate(self, T, seed=42):
        # ogata thinning, see bacry et al 2015
        rng = np.random.RandomState(seed)
        events = [[] for _ in range(self.n_dims)]
        aux = np.zeros((self.n_dims, self.n_dims))
        t = 0.0

        while t < T:
            lambdas = self.mu.copy() + aux.sum(axis=1)
            lambda_bar = lambdas.sum()

            if lambda_bar <= 0:
                lambda_bar = self.mu.sum()

            dt = rng.exponential(1.0 / max(lambda_bar, 1e-10))
            t += dt

            if t >= T:
                break

            for k in range(self.n_dims):
                for l in range(self.n_dims):
                    aux[k, l] *= np.exp(-self.beta[k, l] * dt)

            # TODO: vectorize this loop, it's slow af
            lambdas = self.mu.copy() + aux.sum(axis=1)
            lambda_total = lambdas.sum()

            u = rng.uniform()
            if u <= lambda_total / max(lambda_bar, 1e-10):
                probs = lambdas / max(lambda_total, 1e-10)
                probs = np.maximum(probs, 0)
                probs /= probs.sum()
                k_star = rng.choice(self.n_dims, p=probs)
                events[k_star].append(t)

                for k in range(self.n_dims):
                    aux[k, k_star] += self.alpha[k, k_star] * self.beta[k, k_star]

        return events


def build_hft_hawkes(seed=42):
    # 6-dim hawkes for LOB, see cont 2011 / large 2007
    # 0: bid LO, 1: ask LO, 2: bid cancel, 3: ask cancel, 4: buy MO, 5: sell MO
    n_dims = 6
    mu = np.array([5.0, 5.0, 2.0, 2.0, 1.0, 1.0])

    alpha = np.zeros((n_dims, n_dims))

    alpha[0, 0] = 0.3  # bid limit self-excite
    alpha[1, 1] = 0.3
    alpha[4, 4] = 0.5  # MO momentum
    alpha[5, 5] = 0.5

    alpha[1, 4] = 0.6  # MO triggers liquidity replenishment
    alpha[0, 5] = 0.6

    alpha[2, 3] = 0.4  # cancel contagion
    alpha[3, 2] = 0.4

    alpha[2, 4] = 0.3  # adverse selection
    alpha[3, 5] = 0.3

    beta = np.full((n_dims, n_dims), 10.0)
    beta[4, 4] = 20.0
    beta[5, 5] = 20.0

    return HawkesProcess(n_dims=n_dims, mu=mu, alpha=alpha, beta=beta)


class LOBSimulator:
    def __init__(self, n_levels=5, tick_size=0.01,
                 initial_mid=100.0, base_volume=100.0):
        self.n_levels = n_levels
        self.tick_size = tick_size
        self.initial_mid = initial_mid
        self.base_volume = base_volume

    def generate_snapshots(self, events, T, dt=0.01, seed=42):
        rng = np.random.RandomState(seed)
        n_snapshots = int(T / dt)
        timestamps = np.linspace(0, T, n_snapshots)

        bid_volumes = np.zeros((n_snapshots, self.n_levels))
        ask_volumes = np.zeros((n_snapshots, self.n_levels))
        mid_prices = np.zeros(n_snapshots)

        bid_vol = rng.exponential(self.base_volume, size=self.n_levels)
        ask_vol = rng.exponential(self.base_volume, size=self.n_levels)
        mid = self.initial_mid

        event_ptrs = [0] * 6

        for i, t in enumerate(timestamps):
            for dim in range(6):
                while (event_ptrs[dim] < len(events[dim]) and
                       events[dim][event_ptrs[dim]] <= t):
                    event_ptrs[dim] += 1

                    if dim == 0:
                        level = rng.randint(0, self.n_levels)
                        bid_vol[level] += rng.exponential(20.0)
                    elif dim == 1:
                        level = rng.randint(0, self.n_levels)
                        ask_vol[level] += rng.exponential(20.0)
                    elif dim == 2:
                        level = rng.randint(0, self.n_levels)
                        bid_vol[level] = max(0, bid_vol[level] - rng.exponential(15.0))
                    elif dim == 3:
                        level = rng.randint(0, self.n_levels)
                        ask_vol[level] = max(0, ask_vol[level] - rng.exponential(15.0))
                    elif dim == 4:
                        # buy MO lifts ask
                        ask_vol[0] = max(0, ask_vol[0] - rng.exponential(30.0))
                        if ask_vol[0] < 5.0:
                            mid += self.tick_size
                            ask_vol = np.roll(ask_vol, -1)
                            ask_vol[-1] = rng.exponential(self.base_volume)
                    elif dim == 5:
                        # sell MO hits bid
                        bid_vol[0] = max(0, bid_vol[0] - rng.exponential(30.0))
                        if bid_vol[0] < 5.0:
                            mid -= self.tick_size
                            bid_vol = np.roll(bid_vol, -1)
                            bid_vol[-1] = rng.exponential(self.base_volume)

            if bid_vol[0] < 1.0:
                bid_vol[0] = rng.exponential(self.base_volume * 0.5)
            if ask_vol[0] < 1.0:
                ask_vol[0] = rng.exponential(self.base_volume * 0.5)

            bid_volumes[i] = bid_vol.copy()
            ask_volumes[i] = ask_vol.copy()
            mid_prices[i] = mid

        return timestamps, bid_volumes, ask_volumes, mid_prices


def generate_lob_data(T=10.0, dt=0.01, n_levels=5, seed=42, regime='normal'):
    hawkes = build_hft_hawkes(seed=seed)

    if regime == 'volatile':
        hawkes.mu[4] *= 2.0
        hawkes.mu[5] *= 2.0
        hawkes.alpha[4, 4] = 0.8
        hawkes.alpha[5, 5] = 0.8
    elif regime == 'illiquid':
        hawkes.mu[0] *= 0.3
        hawkes.mu[1] *= 0.3
        hawkes.mu[2] *= 2.0
        hawkes.mu[3] *= 2.0

    events = hawkes.simulate(T=T, seed=seed)
    simulator = LOBSimulator(n_levels=n_levels)
    timestamps, bid_volumes, ask_volumes, mid_prices = simulator.generate_snapshots(
        events, T=T, dt=dt, seed=seed
    )

    return {
        'timestamps': timestamps,
        'bid_volumes': bid_volumes,
        'ask_volumes': ask_volumes,
        'mid_prices': mid_prices,
        'events': events,
        'n_levels': n_levels,
        'regime': regime
    }


def generate_multi_regime_data(T_per_regime=5.0, dt=0.01, n_levels=5, seed=42):
    # normal->volatile->illiquid->normal for drift experiments
    regimes = ['normal', 'volatile', 'illiquid', 'normal']
    all_ts, all_bid, all_ask, all_mid = [], [], [], []
    regime_labels = []
    t_offset = 0.0

    for i, regime in enumerate(regimes):
        data = generate_lob_data(
            T=T_per_regime, dt=dt, n_levels=n_levels,
            seed=seed + i, regime=regime
        )
        all_ts.append(data['timestamps'] + t_offset)
        all_bid.append(data['bid_volumes'])
        all_ask.append(data['ask_volumes'])
        all_mid.append(data['mid_prices'])
        regime_labels.extend([i] * len(data['timestamps']))
        t_offset += T_per_regime

    return {
        'timestamps': np.concatenate(all_ts),
        'bid_volumes': np.concatenate(all_bid),
        'ask_volumes': np.concatenate(all_ask),
        'mid_prices': np.concatenate(all_mid),
        'regime_labels': np.array(regime_labels),
        'regime_names': regimes,
        'n_levels': n_levels
    }


if __name__ == '__main__':
    print("Generating normal LOB data...")
    data = generate_lob_data(T=5.0, seed=42)
    print(f"  Snapshots: {len(data['timestamps'])}")
    print(f"  Bid volume shape: {data['bid_volumes'].shape}")
    print(f"  Ask volume shape: {data['ask_volumes'].shape}")
    print(f"  Mid price range: [{data['mid_prices'].min():.2f}, {data['mid_prices'].max():.2f}]")
    for i, ev in enumerate(data['events']):
        names = ['BidLO', 'AskLO', 'BidCancel', 'AskCancel', 'BuyMO', 'SellMO']
        print(f"  {names[i]} events: {len(ev)}")

    print("\nGenerating multi-regime data...")
    mr_data = generate_multi_regime_data(T_per_regime=3.0, seed=42)
    print(f"  Total snapshots: {len(mr_data['timestamps'])}")
    print(f"  Regime sequence: {mr_data['regime_names']}")
