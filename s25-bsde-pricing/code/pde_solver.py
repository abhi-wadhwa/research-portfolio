import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# CN finite difference solver for nonlinear BS-type PDEs
# see duffy "finite difference methods in financial engineering"


def build_grid(S_max, N_S, T, N_t):
    # uniform grid in (S, t)
    S = np.linspace(0, S_max, N_S + 1)
    t = np.linspace(0, T, N_t + 1)
    dS = S[1] - S[0]
    dt = t[1] - t[0]
    return S, t, dS, dt


def solve_linear_bs_cn(S0, K, r, sigma, T, N_S=200, N_t=500,
                        S_max_mult=4.0, option_type='call'):
    # crank-nicolson for linear BS PDE
    # works but slow for large grids, could use SOR
    S_max = S_max_mult * K
    S, t, dS, dt = build_grid(S_max, N_S, T, N_t)
    theta = 0.5  # CN

    if option_type == 'call':
        U_terminal = np.maximum(S - K, 0.0)
    else:
        U_terminal = np.maximum(K - S, 0.0)

    U = np.zeros((N_t + 1, N_S + 1))
    U[-1, :] = U_terminal

    n = N_S - 1  # interior pts

    j = np.arange(1, N_S)
    Sj = S[j]

    alpha = 0.5 * sigma**2 * Sj**2 / dS**2
    beta_coeff = r * Sj / (2 * dS)

    a = -alpha + beta_coeff   # sub-diag
    b = 2 * alpha + r         # diag
    c = -alpha - beta_coeff   # super-diag

    # implicit side
    diag_imp = 1 + theta * dt * b
    lower_imp = theta * dt * a[1:]
    upper_imp = theta * dt * c[:-1]
    A_imp = sparse.diags([lower_imp, diag_imp, upper_imp], [-1, 0, 1],
                          shape=(n, n), format='csc')

    # explicit side
    diag_exp = 1 - (1 - theta) * dt * b
    lower_exp = -(1 - theta) * dt * a[1:]
    upper_exp = -(1 - theta) * dt * c[:-1]
    A_exp = sparse.diags([lower_exp, diag_exp, upper_exp], [-1, 0, 1],
                          shape=(n, n), format='csc')

    # march backward
    for k in range(N_t - 1, -1, -1):
        u_old = U[k + 1, 1:N_S]

        rhs = A_exp @ u_old
        # BCs
        bc_low = 0.0
        if option_type == 'put':
            bc_low = K * np.exp(-r * (T - t[k]))
        bc_high = 0.0
        if option_type == 'call':
            bc_high = S_max - K * np.exp(-r * (T - t[k]))

        rhs[0] -= (theta * dt * a[0] + (1 - theta) * dt * a[0]) * 0
        rhs[0] += theta * dt * (-a[0]) * bc_low + (1 - theta) * dt * (-a[0]) * bc_low
        rhs[-1] += theta * dt * (-c[-1]) * bc_high + (1 - theta) * dt * (-c[-1]) * bc_high

        U[k, 1:N_S] = spsolve(A_imp, rhs)
        U[k, 0] = bc_low
        U[k, N_S] = bc_high

    price_at_S0 = np.interp(S0, S, U[0, :])
    return S, t, U, price_at_S0


def solve_nonlinear_pde_cn(S0, K, r, sigma_base, T, nonlinear_term,
                             N_S=200, N_t=500, S_max_mult=4.0,
                             option_type='call', picard_iter=5):
    # CN + picard iteration for nonlinear terms
    # picard converges if dt small enough (contraction mapping)
    # TODO: clean up - could refactor with solve_linear_bs_cn
    S_max = S_max_mult * K
    S, t_grid, dS, dt = build_grid(S_max, N_S, T, N_t)
    theta = 0.5
    n = N_S - 1

    if option_type == 'call':
        U_terminal = np.maximum(S - K, 0.0)
    else:
        U_terminal = np.maximum(K - S, 0.0)

    U = np.zeros((N_t + 1, N_S + 1))
    U[-1, :] = U_terminal

    j = np.arange(1, N_S)
    Sj = S[j]
    sigma = sigma_base

    alpha = 0.5 * sigma**2 * Sj**2 / dS**2
    beta_coeff = r * Sj / (2 * dS)
    a = -alpha + beta_coeff
    b = 2 * alpha + r
    c = -alpha - beta_coeff

    diag_imp = 1 + theta * dt * b
    lower_imp = theta * dt * a[1:]
    upper_imp = theta * dt * c[:-1]
    A_imp = sparse.diags([lower_imp, diag_imp, upper_imp], [-1, 0, 1],
                          shape=(n, n), format='csc')

    diag_exp = 1 - (1 - theta) * dt * b
    lower_exp = -(1 - theta) * dt * a[1:]
    upper_exp = -(1 - theta) * dt * c[:-1]
    A_exp = sparse.diags([lower_exp, diag_exp, upper_exp], [-1, 0, 1],
                          shape=(n, n), format='csc')

    for k in range(N_t - 1, -1, -1):
        u_old = U[k + 1, 1:N_S]

        bc_low = 0.0
        bc_high = 0.0
        if option_type == 'call':
            bc_high = S_max - K * np.exp(-r * (T - t_grid[k]))
        elif option_type == 'put':
            bc_low = K * np.exp(-r * (T - t_grid[k]))

        rhs_base = A_exp @ u_old
        rhs_base[0] += theta * dt * (-a[0]) * bc_low + (1 - theta) * dt * (-a[0]) * bc_low
        rhs_base[-1] += theta * dt * (-c[-1]) * bc_high + (1 - theta) * dt * (-c[-1]) * bc_high

        # picard loop
        u_guess = u_old.copy()
        for _ in range(picard_iter):
            u_full = np.zeros(N_S + 1)
            u_full[1:N_S] = u_guess
            u_full[0] = bc_low
            u_full[N_S] = bc_high
            u_S_deriv = (u_full[2:N_S + 1] - u_full[0:N_S - 1]) / (2 * dS)  # central diff
            z_vals = Sj * u_S_deriv  # delta hedge term

            h_vals = nonlinear_term(Sj, u_guess, z_vals)
            rhs = rhs_base + dt * h_vals
            u_guess = spsolve(A_imp, rhs)

        U[k, 1:N_S] = u_guess
        U[k, 0] = bc_low
        U[k, N_S] = bc_high

    price_at_S0 = np.interp(S0, S, U[0, :])
    return S, t_grid, U, price_at_S0


# nonlinear term builders

def transaction_cost_nl(r, kappa_tc):
    # h = -kappa*|z| (proportional tc)
    def h(S_arr, u_arr, z_arr):
        return -kappa_tc * np.abs(z_arr)
    return h


def funding_cost_nl(r_lend, r_borrow, r_base):
    # h = -(R(u)-r_base)*u, R switches on sign of u
    def h(S_arr, u_arr, z_arr):
        rate = np.where(u_arr >= 0, r_lend, r_borrow)
        return -(rate - r_base) * u_arr
    return h


def uncertain_vol_nl(sigma_low, sigma_high, sigma_base):
    # worst-case vol via gamma sign
    # this is janky - using |z| as proxy when we don't have gamma directly
    # h = 0.5*(sigma_H^2 - sigma_L^2)*|z|
    lam = 0.5 * (sigma_high**2 - sigma_low**2)
    def h(S_arr, u_arr, z_arr):
        return lam * np.abs(z_arr)
    return h
