import numpy as np
from numpy.polynomial.polynomial import polyvander

# g-expectations (peng 1997/2004/2019)
# -dY_t = g(t,Z_t)dt - Z_t dW_t, Y_T = xi
# mapping xi -> Y_t is cond g-exp E^g_t[xi]
#
# properties to verify:
#   time-consistency, monotonicity, translation invariance,
#   subadditivity (g sublinear), positive homogeneity


def solve_g_bsde(X, t, terminal_values, g_driver, basis_degree=4):
    # same as bsde_solver but driver only depends on z
    M, Np1 = X.shape
    N = Np1 - 1
    dt_arr = np.diff(t)

    Y = np.zeros((M, N + 1))
    Z = np.zeros((M, N + 1))
    Y[:, N] = terminal_values

    for i in range(N - 1, -1, -1):
        dt = dt_arr[i]
        dX = X[:, i + 1] - X[:, i]

        xn = (X[:, i] - np.mean(X[:, i])) / (np.std(X[:, i]) + 1e-12)
        basis = polyvander(xn, basis_degree)

        # Z regression
        target_z = Y[:, i + 1] * dX / (np.maximum(np.abs(X[:, i]), 1e-8) * dt)
        coeffs_z, _, _, _ = np.linalg.lstsq(basis, target_z, rcond=None)
        Z[:, i] = basis @ coeffs_z

        # backward euler
        rhs = Y[:, i + 1] + g_driver(t[i], Z[:, i]) * dt
        coeffs_y, _, _, _ = np.linalg.lstsq(basis, rhs, rcond=None)
        Y[:, i] = basis @ coeffs_y

    return Y, Z


# g-drivers

def sublinear_g(gamma):
    # g(z) = gamma*|z| -> coherent risk measure
    # corresponds to worst-case drift ambiguity bounded by gamma
    # see zhang ch.4
    def g(t, z):
        return gamma * np.abs(z)
    return g


def quadratic_g(gamma):
    # g(z) = gamma*z^2 -> convex but not coherent
    def g(t, z):
        return gamma * z ** 2
    return g


def zero_g():
    # g=0 -> reduces to cond exp E[xi|F_t]
    def g(t, z):
        return np.zeros_like(z)
    return g


# entropic risk measure for comparison
# rho(xi) = (1/theta)*ln E[exp(-theta*xi)]

def entropic_risk_measure(xi, theta):
    # log-sum-exp trick
    max_val = np.max(-theta * xi)
    log_E = max_val + np.log(np.mean(np.exp(-theta * xi - max_val)))
    return log_E / theta


def conditional_entropic(X, t, terminal_values, theta, basis_degree=4):
    # backward recursion approx for conditional entropic risk
    # rho_t = (1/theta)*ln E[exp(-theta*xi)|F_t]
    # this is janky - regression on exp(theta*rho) is numerically iffy
    M, Np1 = X.shape
    N = Np1 - 1
    dt_arr = np.diff(t)

    rho = np.zeros((M, N + 1))
    rho[:, N] = -terminal_values  # sign convention

    for i in range(N - 1, -1, -1):
        xn = (X[:, i] - np.mean(X[:, i])) / (np.std(X[:, i]) + 1e-12)
        basis = polyvander(xn, basis_degree)

        exp_val = np.exp(np.clip(theta * rho[:, i + 1], -50, 50))
        coeffs, _, _, _ = np.linalg.lstsq(basis, exp_val, rcond=None)
        cond_exp = np.maximum(basis @ coeffs, 1e-30)
        rho[:, i] = np.log(cond_exp) / theta

    return rho


# time-consistency verification: E^g_s[E^g_t[xi]] = E^g_s[xi]

def verify_time_consistency(X, t, terminal_values, g_driver, s_idx, t_idx,
                            basis_degree=4, n_trials=5):
    # direct: full BSDE from T to s
    Y_direct, _ = solve_g_bsde(X, t, terminal_values, g_driver,
                                basis_degree=basis_degree)
    direct_val = np.mean(Y_direct[:, s_idx])

    # composed: T->t then t->s
    Y_full, _ = solve_g_bsde(X, t, terminal_values, g_driver,
                              basis_degree=basis_degree)
    inner_values = Y_full[:, t_idx]

    # now solve sub-BSDE from t_idx back to s_idx
    X_sub = X[:, :t_idx + 1]
    t_sub = t[:t_idx + 1]
    Y_outer, _ = solve_g_bsde(X_sub, t_sub, inner_values, g_driver,
                               basis_degree=basis_degree)
    composed_val = np.mean(Y_outer[:, s_idx])

    return direct_val, composed_val


# full analysis

def run_g_expectation_analysis(S0=100.0, mu=0.05, sigma=0.2, T=1.0,
                                N=100, M=50000, seed=42):
    from sde_simulation import gbm_euler_maruyama

    rng = np.random.default_rng(seed)
    t_grid, X = gbm_euler_maruyama(S0, mu, sigma, T, N, M, rng=rng)

    K = 100.0
    xi = np.maximum(X[:, -1] - K, 0.0)  # call payoff

    # linear expectation (g=0)
    print("Computing cond exp (g=0)...")
    Y_linear, Z_linear = solve_g_bsde(X, t_grid, xi, zero_g(), basis_degree=4)

    # g-exp for various gamma
    gammas = [0.0, 0.05, 0.1, 0.2, 0.5]
    g_results = {}
    for gamma in gammas:
        print(f"Computing g-exp (gamma={gamma})...")
        Y_g, Z_g = solve_g_bsde(X, t_grid, xi, sublinear_g(gamma),
                                  basis_degree=4)
        g_results[gamma] = {'Y': Y_g, 'Z': Z_g, 'Y0': np.mean(Y_g[:, 0])}

    # entropic comparison
    print("Computing entropic risk measure...")
    thetas = [0.001, 0.005, 0.01]
    entropic_results = {}
    for theta in thetas:
        rho_val = entropic_risk_measure(xi, theta)
        entropic_results[theta] = rho_val

    # time-consistency check
    print("Verifying time-consistency...")
    s_idx = 25   # t = T/4
    t_idx = 50   # t = T/2
    tc_results = {}
    for gamma in [0.1, 0.2]:
        direct, composed = verify_time_consistency(
            X, t_grid, xi, sublinear_g(gamma), s_idx, t_idx, basis_degree=4
        )
        tc_results[gamma] = {
            'direct': direct, 'composed': composed,
            'error': abs(direct - composed),
            'rel_error': abs(direct - composed) / (abs(direct) + 1e-12),
        }
        print(f"  gamma={gamma}: direct={direct:.4f}, composed={composed:.4f}, "
              f"rel_error={tc_results[gamma]['rel_error']:.6f}")

    # time-consistency across multiple t
    print("Computing time-consistency across time points...")
    gamma_tc = 0.1
    tc_time_points = []
    t_indices = list(range(10, 90, 10))
    for ti in t_indices:
        direct, composed = verify_time_consistency(
            X, t_grid, xi, sublinear_g(gamma_tc), 0, ti, basis_degree=4
        )
        tc_time_points.append({
            'time': t_grid[ti],
            'direct': direct, 'composed': composed,
            'error': abs(direct - composed),
        })

    results = {
        't_grid': t_grid,
        'X': X,
        'xi': xi,
        'Y_linear': Y_linear,
        'g_results': g_results,
        'gammas': gammas,
        'entropic_results': entropic_results,
        'tc_results': tc_results,
        'tc_time_points': tc_time_points,
        'tc_gamma': gamma_tc,
    }
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("g-Expectation Analysis")
    print("=" * 60)
    results = run_g_expectation_analysis(M=30000, N=100)

    print("\n--- g-exp values at t=0 ---")
    for gamma, data in results['g_results'].items():
        print(f"  gamma={gamma}: Y_0 = {data['Y0']:.4f}")

    print("\n--- entropic risk measures ---")
    for theta, rho in results['entropic_results'].items():
        print(f"  theta={theta}: rho = {rho:.4f}")
