import numpy as np
from numpy.polynomial.polynomial import polyvander

# tried picard iteration first but regression is way faster for this
# see bouchard-touzi 2004, zhang 2004, gobet et al 2005

def polynomial_basis(x, degree=4):
    # poly basis for LS regression, normalize for stability
    xn = (x - np.mean(x)) / (np.std(x) + 1e-12)
    return polyvander(xn, degree)


def solve_bsde(X, t, terminal_condition, driver, r=0.0, basis_degree=4):
    # backward induction w/ least-squares regression
    M, Np1 = X.shape
    N = Np1 - 1
    dt_arr = np.diff(t)

    Y = np.zeros((M, N + 1))
    Z = np.zeros((M, N + 1))

    # terminal cond
    Y[:, N] = terminal_condition(X[:, N])

    # old approach: reconstruct dW from paths directly
    # didn't work well, switched to regression on dX instead

    for i in range(N - 1, -1, -1):
        dt = dt_arr[i]
        dX = X[:, i + 1] - X[:, i]

        basis = polynomial_basis(X[:, i], degree=basis_degree)

        # Z regression - zhang (2004) formula
        # hack: using dX/(X*dt) as proxy for dW/dt, works for diffusion ~ sigma*X
        target_z = Y[:, i + 1] * dX / (np.maximum(np.abs(X[:, i]), 1e-8) * dt)
        coeffs_z, _, _, _ = np.linalg.lstsq(basis, target_z, rcond=None)
        Z[:, i] = basis @ coeffs_z

        # backward euler step
        rhs = Y[:, i + 1] + driver(t[i], Y[:, i + 1], Z[:, i]) * dt

        coeffs_y, _, _, _ = np.linalg.lstsq(basis, rhs, rcond=None)
        Y[:, i] = basis @ coeffs_y

    return Y, Z


# payoffs

def european_call_payoff(K):
    def payoff(x):
        return np.maximum(x - K, 0.0)
    return payoff


def european_put_payoff(K):
    def payoff(x):
        return np.maximum(K - x, 0.0)
    return payoff

# tried asian payoff too but need running avg state var, TODO
# def asian_call_payoff(K):
#     def payoff(x_avg):
#         return np.maximum(x_avg - K, 0.0)
#     return payoff


# drivers

def linear_driver(r):
    # f(t,y,z) = -r*y (standard BS)
    def f(t, y, z):
        return -r * y
    return f


def transaction_cost_driver(r, kappa_tc):
    # leland 1985 style, |z| term = hedging cost from gamma trading
    def f(t, y, z):
        return -r * y - kappa_tc * np.abs(z)
    return f


def funding_cost_driver(r_lend, r_borrow):
    # bergman 1995 - different borrow/lend rates
    def f(t, y, z):
        rate = np.where(y >= 0, r_lend, r_borrow)
        return -rate * y
    return f


def uncertain_vol_driver(r, sigma_low, sigma_high):
    # avellaneda et al 1995 worst-case pricing
    # simplified: f = -ry + lambda*|z| where lambda encodes vol band
    lam = 0.5 * (sigma_high**2 - sigma_low**2)
    def f(t, y, z):
        return -r * y + lam * np.abs(z)
    return f
