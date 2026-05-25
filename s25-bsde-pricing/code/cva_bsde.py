import numpy as np

# CVA via BSDE
# unilateral CVA on IRS netting set, vasicek short rate, const hazard rate
# see bichuch, capponi & sturm (2018), crepey et al (2014)
#
# BSDE: -dV_t = [-r_t*V_t - lambda*LGD*max(V_t,0)]dt - Z_t dW_t


# vasicek: dr_t = kappa*(theta - r_t)dt + sigma_r dW_t

def simulate_vasicek(r0, kappa, theta, sigma_r, T, N, M, rng=None):
    # euler step for vasicek short rate
    if rng is None:
        rng = np.random.default_rng(42)
    dt = T / N
    t = np.linspace(0, T, N + 1)
    r = np.zeros((M, N + 1))
    r[:, 0] = r0

    for i in range(N):
        dW = rng.standard_normal(M) * np.sqrt(dt)
        r[:, i + 1] = r[:, i] + kappa * (theta - r[:, i]) * dt + sigma_r * dW

    return t, r


def vasicek_bond_price(r, t, T_mat, kappa, theta, sigma_r):
    # analytical ZCB price P(t,T) = exp(A-B*r)
    tau = T_mat - t
    B = (1.0 - np.exp(-kappa * tau)) / kappa
    A = (theta - 0.5 * sigma_r ** 2 / kappa ** 2) * (B - tau) \
        - 0.25 * sigma_r ** 2 * B ** 2 / kappa
    return np.exp(A - B * r)


# portfolio of vanilla IRS

def generate_swap_portfolio(n_swaps=5, rng=None):
    if rng is None:
        rng = np.random.default_rng(123)

    maturities = rng.choice([2, 3, 5, 7, 10], size=n_swaps, replace=True).astype(float)
    fixed_rates = 0.02 + 0.03 * rng.random(n_swaps)
    notionals = 1e6 * (1 + rng.integers(1, 10, size=n_swaps))
    pay_fixed = rng.choice([True, False], size=n_swaps)

    swaps = []
    for i in range(n_swaps):
        swaps.append({
            'notional': notionals[i],
            'fixed_rate': fixed_rates[i],
            'maturity': maturities[i],
            'pay_fixed': pay_fixed[i],
        })
    return swaps


def swap_mtm(swap, r_t, t, kappa, theta, sigma_r):
    # simplified swap MTM - not exact but good enough for demo
    # TODO: proper annuity factor calculation
    T_mat = swap['maturity']
    if t >= T_mat:
        return np.zeros_like(r_t)

    tau = T_mat - t
    P = vasicek_bond_price(r_t, t, T_mat, kappa, theta, sigma_r)

    fixed_pv = swap['notional'] * swap['fixed_rate'] * tau * (1 + P) / 2
    float_pv = swap['notional'] * (1.0 - P)

    if swap['pay_fixed']:
        value = float_pv - fixed_pv
    else:
        value = fixed_pv - float_pv

    return value


def portfolio_mtm(swaps, r_t, t, kappa, theta, sigma_r):
    # netting set = sum of swap MTMs
    V = np.zeros_like(r_t)
    for swap in swaps:
        V += swap_mtm(swap, r_t, t, kappa, theta, sigma_r)
    return V


# exposure profiles

def compute_exposure_profiles(swaps, t_grid, r_paths, kappa, theta, sigma_r):
    # EE = E[max(V_t,0)], EPE = time avg of EE
    M = r_paths.shape[0]
    N = len(t_grid) - 1
    V_paths = np.zeros((M, N + 1))

    for i, t in enumerate(t_grid):
        V_paths[:, i] = portfolio_mtm(swaps, r_paths[:, i], t, kappa, theta, sigma_r)

    positive_exposure = np.maximum(V_paths, 0.0)
    EE = np.mean(positive_exposure, axis=0)

    dt_vals = np.diff(t_grid)
    EPE = np.sum(0.5 * (EE[:-1] + EE[1:]) * dt_vals) / t_grid[-1]

    return EE, EPE, V_paths


# MC CVA (benchmark)

def cva_monte_carlo(swaps, t_grid, r_paths, kappa, theta, sigma_r,
                    lambda_cpty=0.02, recovery=0.4):
    # CVA = LGD * int_0^T lambda * DF(0,t) * EE(t) dt
    LGD = 1.0 - recovery
    EE, EPE, V_paths = compute_exposure_profiles(
        swaps, t_grid, r_paths, kappa, theta, sigma_r
    )

    # discount using mean short rate (not great but ok for comparison)
    r_mean = np.mean(r_paths, axis=0)
    dt_vals = np.diff(t_grid)
    cum_r = np.cumsum(np.concatenate([[0], r_mean[:-1] * dt_vals]))
    DF = np.exp(-cum_r)

    integrand = LGD * lambda_cpty * DF * EE
    cva_profile = np.zeros(len(t_grid))
    for i in range(1, len(t_grid)):
        cva_profile[i] = cva_profile[i - 1] + \
                         0.5 * (integrand[i - 1] + integrand[i]) * dt_vals[i - 1]

    cva = cva_profile[-1]
    return cva, cva_profile, EE, DF


# BSDE CVA

def cva_bsde(swaps, t_grid, r_paths, kappa, theta, sigma_r,
             lambda_cpty=0.02, recovery=0.4, basis_degree=3):
    # regression-based BSDE for CVA-adjusted value
    # driver: f(t,y,z) = -r_t*y - lambda*LGD*max(y,0)
    from numpy.polynomial.polynomial import polyvander

    LGD = 1.0 - recovery
    M, Np1 = r_paths.shape
    N = Np1 - 1
    dt_arr = np.diff(t_grid)

    V_riskfree = np.zeros((M, N + 1))
    for i, t in enumerate(t_grid):
        V_riskfree[:, i] = portfolio_mtm(swaps, r_paths[:, i], t,
                                          kappa, theta, sigma_r)

    Y = np.zeros((M, N + 1))
    Z = np.zeros((M, N + 1))

    Y[:, N] = V_riskfree[:, N]

    for i in range(N - 1, -1, -1):
        dt = dt_arr[i]

        # 2d basis on (r_t, V_t)
        x1 = r_paths[:, i]
        x2 = V_riskfree[:, i]
        x1_n = (x1 - np.mean(x1)) / (np.std(x1) + 1e-12)
        x2_n = (x2 - np.mean(x2)) / (np.std(x2) + 1e-12)

        basis_1d = polyvander(x1_n, basis_degree)
        basis_2d = polyvander(x2_n, min(basis_degree, 2))
        basis = np.column_stack([basis_1d, basis_2d[:, 1:]])  # skip dup constant

        # driver
        def driver(y_val):
            return -r_paths[:, i] * y_val - lambda_cpty * LGD * np.maximum(y_val, 0)

        rhs = Y[:, i + 1] + driver(Y[:, i + 1]) * dt

        coeffs, _, _, _ = np.linalg.lstsq(basis, rhs, rcond=None)
        Y[:, i] = basis @ coeffs

    # CVA = risk-free - adjusted
    cva = np.mean(V_riskfree[:, 0]) - np.mean(Y[:, 0])

    return cva, Y, V_riskfree


def run_cva_analysis(M=20000, N=100, T=10.0, n_swaps=7,
                     r0=0.03, kappa=0.5, theta=0.04, sigma_r=0.01,
                     lambda_cpty=0.02, recovery=0.4, seed=42):
    # full CVA pipeline: exposures, MC CVA, BSDE CVA
    rng = np.random.default_rng(seed)
    swaps = generate_swap_portfolio(n_swaps=n_swaps, rng=rng)

    rng_sim = np.random.default_rng(seed + 1)
    t_grid, r_paths = simulate_vasicek(r0, kappa, theta, sigma_r, T, N, M, rng=rng_sim)

    EE, EPE, V_paths = compute_exposure_profiles(
        swaps, t_grid, r_paths, kappa, theta, sigma_r
    )

    cva_mc, cva_mc_profile, _, DF = cva_monte_carlo(
        swaps, t_grid, r_paths, kappa, theta, sigma_r,
        lambda_cpty=lambda_cpty, recovery=recovery
    )

    cva_bsde_val, Y_bsde, V_riskfree = cva_bsde(
        swaps, t_grid, r_paths, kappa, theta, sigma_r,
        lambda_cpty=lambda_cpty, recovery=recovery
    )

    print(f"CVA (Monte Carlo): {cva_mc:,.2f}")
    print(f"CVA (BSDE):        {cva_bsde_val:,.2f}")
    print(f"EPE:               {EPE:,.2f}")

    return {
        't_grid': t_grid,
        'r_paths': r_paths,
        'swaps': swaps,
        'EE': EE,
        'EPE': EPE,
        'V_paths': V_paths,
        'cva_mc': cva_mc,
        'cva_mc_profile': cva_mc_profile,
        'cva_bsde': cva_bsde_val,
        'Y_bsde': Y_bsde,
        'V_riskfree': V_riskfree,
        'DF': DF,
        'lambda_cpty': lambda_cpty,
        'recovery': recovery,
        'kappa': kappa,
        'theta': theta,
        'sigma_r': sigma_r,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("CVA Analysis via BSDE")
    print("=" * 60)
    results = run_cva_analysis()
