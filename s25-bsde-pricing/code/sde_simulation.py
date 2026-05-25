import numpy as np


# GBM stuff

def gbm_euler_maruyama(S0, mu, sigma, T, N, M, rng=None):
    # basic EM for geometric BM
    if rng is None:
        rng = np.random.default_rng(42)
    dt = T / N
    t = np.linspace(0, T, N + 1)
    S = np.zeros((M, N + 1))
    S[:, 0] = S0
    for i in range(N):
        dW = rng.standard_normal(M) * np.sqrt(dt)
        S[:, i + 1] = S[:, i] + mu * S[:, i] * dt + sigma * S[:, i] * dW
        S[:, i + 1] = np.maximum(S[:, i + 1], 1e-8)  # absorb at 0
    return t, S


def gbm_milstein(S0, mu, sigma, T, N, M, rng=None):
    # milstein for GBM - strong order 1
    # correction term: 0.5*sigma^2*S*(dW^2 - dt), see zhang ch.4
    if rng is None:
        rng = np.random.default_rng(42)
    dt = T / N
    t = np.linspace(0, T, N + 1)
    S = np.zeros((M, N + 1))
    S[:, 0] = S0
    for i in range(N):
        dW = rng.standard_normal(M) * np.sqrt(dt)
        S[:, i + 1] = (S[:, i]
                        + mu * S[:, i] * dt
                        + sigma * S[:, i] * dW
                        + 0.5 * sigma**2 * S[:, i] * (dW**2 - dt))
        S[:, i + 1] = np.maximum(S[:, i + 1], 1e-8)
    return t, S


def gbm_exact(S0, mu, sigma, T, N, M, rng=None):
    # exact log-normal sim for benchmarking
    if rng is None:
        rng = np.random.default_rng(42)
    dt = T / N
    t = np.linspace(0, T, N + 1)
    S = np.zeros((M, N + 1))
    S[:, 0] = S0
    for i in range(N):
        Z = rng.standard_normal(M)
        S[:, i + 1] = S[:, i] * np.exp((mu - 0.5 * sigma**2) * dt
                                         + sigma * np.sqrt(dt) * Z)
    return t, S


# CEV model dS = mu*S dt + sigma*S^gamma dW

def cev_euler_maruyama(S0, mu, sigma, gamma, T, N, M, rng=None):
    # EM for CEV
    if rng is None:
        rng = np.random.default_rng(42)
    dt = T / N
    t = np.linspace(0, T, N + 1)
    S = np.zeros((M, N + 1))
    S[:, 0] = S0
    for i in range(N):
        dW = rng.standard_normal(M) * np.sqrt(dt)
        S[:, i + 1] = (S[:, i]
                        + mu * S[:, i] * dt
                        + sigma * np.power(np.maximum(S[:, i], 1e-8), gamma) * dW)
        S[:, i + 1] = np.maximum(S[:, i + 1], 1e-8)
    return t, S


def cev_milstein(S0, mu, sigma, gamma, T, N, M, rng=None):
    # milstein for CEV - diffusion b(S)=sigma*S^gamma, b'(S)=sigma*gamma*S^{gamma-1}
    if rng is None:
        rng = np.random.default_rng(42)
    dt = T / N
    t = np.linspace(0, T, N + 1)
    S = np.zeros((M, N + 1))
    S[:, 0] = S0
    for i in range(N):
        dW = rng.standard_normal(M) * np.sqrt(dt)
        Sc = np.maximum(S[:, i], 1e-8)
        b = sigma * np.power(Sc, gamma)
        bp = sigma * gamma * np.power(Sc, gamma - 1)
        S[:, i + 1] = (S[:, i]
                        + mu * S[:, i] * dt
                        + b * dW
                        + 0.5 * b * bp * (dW**2 - dt))
        S[:, i + 1] = np.maximum(S[:, i + 1], 1e-8)
    return t, S


# Heston stochastic vol
# dS = mu*S dt + sqrt(v)*S dW_1
# dv = kappa*(theta - v) dt + xi*sqrt(v) dW_2, corr = rho

def heston_euler_maruyama(S0, v0, mu, kappa, theta, xi, rho, T, N, M, rng=None):
    # EM w/ full truncation scheme for heston
    # TODO: try QE scheme from andersen (2008)
    if rng is None:
        rng = np.random.default_rng(42)
    dt = T / N
    t = np.linspace(0, T, N + 1)
    S = np.zeros((M, N + 1))
    v = np.zeros((M, N + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    for i in range(N):
        Z1 = rng.standard_normal(M)
        Z2 = rng.standard_normal(M)
        W1 = Z1 * np.sqrt(dt)
        W2 = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)  # cholesky
        vp = np.maximum(v[:, i], 0)  # truncation
        S[:, i + 1] = S[:, i] + mu * S[:, i] * dt + np.sqrt(vp) * S[:, i] * W1
        v[:, i + 1] = v[:, i] + kappa * (theta - vp) * dt + xi * np.sqrt(vp) * W2
        S[:, i + 1] = np.maximum(S[:, i + 1], 1e-8)
    return t, S, v


# convergence helpers

def strong_error(S_approx, S_exact):
    # E[|S_approx(T) - S_exact(T)|]
    return np.mean(np.abs(S_approx[:, -1] - S_exact[:, -1]))


def weak_error(S_approx, S_exact):
    # |E[S_approx(T)] - E[S_exact(T)]|
    return np.abs(np.mean(S_approx[:, -1]) - np.mean(S_exact[:, -1]))
