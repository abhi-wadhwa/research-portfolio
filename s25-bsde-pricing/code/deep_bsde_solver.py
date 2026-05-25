import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from han-jentzen-e 2018
# one feedforward net per timestep approximates Z, Y_0 is trainable
# loss = terminal condition mismatch


class SubNet(nn.Module):
    # small net R^d -> R^d, batchnorm helps a lot
    def __init__(self, d_in, d_out, hidden_dim=64, num_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.BatchNorm1d(d_in))
        prev = d_in
        for _ in range(num_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev = hidden_dim
        layers.append(nn.Linear(prev, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepBSDE(nn.Module):
    # one subnet per timestep for Z, Y_0 is a learnable param
    def __init__(self, dim, T, N, driver, terminal, sigma_val=1.0, mu_val=0.0,
                 hidden_dim=64, num_layers=2):
        super().__init__()
        self.dim = dim
        self.T = T
        self.N = N
        self.dt = T / N
        self.driver = driver
        self.terminal = terminal
        self.sigma_val = sigma_val
        self.mu_val = mu_val

        self.y0 = nn.Parameter(torch.tensor([0.5]))  # init guess

        self.z_nets = nn.ModuleList([
            SubNet(dim, dim, hidden_dim=hidden_dim, num_layers=num_layers)
            for _ in range(N)
        ])

    def forward(self, dW, x0=None):
        # roll out the discretized BSDE
        batch = dW.shape[0]
        dt = self.dt
        sqrt_dt = np.sqrt(dt)

        if x0 is None:
            x = torch.zeros(batch, self.dim)
        else:
            x = x0.clone()

        y = self.y0.expand(batch)

        for n in range(self.N):
            dw = dW[:, n, :]

            z = self.z_nets[n](x)  # Z_n = net_n(X_n)

            f_val = self.driver(n * dt, x, y, z)
            f_val = torch.clamp(f_val, -100.0, 100.0)  # stability hack

            # euler step: Y_{n+1} = Y_n - f*dt + Z^T * sigma * dW
            y = y - f_val * dt + self.sigma_val * torch.sum(z * dw, dim=1)
            y = torch.clamp(y, -50.0, 50.0)  # prevent blowup

            # X update
            x = x + self.mu_val * dt + self.sigma_val * dw

        g_val = self.terminal(x)
        return y, g_val


def train_deep_bsde(model, n_epochs=1000, batch_size=256, lr=1e-3,
                    x0_value=None, verbose=True):
    # higher lr for Y_0 seems to help
    y0_params = [model.y0]
    net_params = [p for name, p in model.named_parameters() if 'y0' not in name]
    optimizer = optim.Adam([
        {'params': y0_params, 'lr': lr * 10},
        {'params': net_params, 'lr': lr},
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(n_epochs // 3, 1),
                                          gamma=0.5)

    losses = []
    y0_history = []
    sqrt_dt = np.sqrt(model.dt)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        dW = torch.randn(batch_size, model.N, model.dim) * sqrt_dt

        if x0_value is not None:
            x0 = torch.full((batch_size, model.dim), x0_value)
        else:
            x0 = None

        y_pred, g_true = model(dW, x0)
        loss = torch.mean((y_pred - g_true) ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        y0_history.append(model.y0.item())

        if verbose and (epoch % max(n_epochs // 10, 1) == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:5d}/{n_epochs}: loss={loss.item():.6f}, "
                  f"Y_0={model.y0.item():.6f}")

    return losses, y0_history


# benchmark problems

# 1. Black-Scholes / heat eqn
# u_t + 0.5*Tr(sigma^2 D^2 u) = 0, u(T,x)=||x||^2
# exact: u(t,x) = ||x||^2 + d*sigma^2*(T-t)

def bs_driver(t, x, y, z):
    return torch.zeros(x.shape[0])

def bs_terminal(x):
    return torch.sum(x ** 2, dim=1)

def bs_exact(t, x, sigma, T, d):
    return np.sum(x ** 2) + d * sigma ** 2 * (T - t)


# 2. Allen-Cahn: u_t + 0.5*Tr(...) + u - u^3 = 0
# driver: f = y - y^3

def allen_cahn_driver(t, x, y, z):
    y_clamped = torch.clamp(y, -5.0, 5.0)  # clamp early on
    return y_clamped - y_clamped ** 3

def allen_cahn_terminal(x):
    # g(x) = 1/(2 + 0.4*||x||^2)
    return 1.0 / (2.0 + 0.4 * torch.sum(x ** 2, dim=1))


# 3. HJB: u_t + 0.5*Tr(...) - ||Du||^2 = 0
# driver: f = -||z||^2/sigma^2 (since Z=sigma*Du)

def make_hjb_driver(sigma_val):
    def hjb_driver(t, x, y, z):
        return -torch.sum(z ** 2, dim=1) / (sigma_val ** 2)
    return hjb_driver

def hjb_terminal(x):
    # g(x) = ln(0.5*(1+||x||^2))
    return torch.log(0.5 * (1.0 + torch.sum(x ** 2, dim=1)))


# 4. Bergman diff rates
# dY = -(r*Y - (R-r)*max(0, Y-sum(Z)/sigma))dt + ZdW

def make_bergman_driver(r_lend, r_borrow, sigma_val):
    spread = r_borrow - r_lend
    def bergman_driver(t, x, y, z):
        portfolio_val = y - torch.sum(z, dim=1) / sigma_val
        return -r_lend * y + spread * torch.clamp(portfolio_val, min=0.0)
    return bergman_driver

def bergman_terminal(x):
    # rainbow call on max
    max_val, _ = torch.max(x, dim=1)
    return torch.clamp(max_val - 1.0, min=0.0)


def run_benchmark(name, dim, T=1.0, N=20, sigma_val=1.0, mu_val=0.0,
                  n_epochs=500, batch_size=256, lr=1e-3, hidden_dim=64,
                  num_layers=2, x0_value=0.0):
    # pick a benchmark and train
    if name == 'black_scholes':
        driver = bs_driver
        terminal = bs_terminal
        exact_y0 = bs_exact(0.0, np.full(dim, x0_value), sigma_val, T, dim)
    elif name == 'allen_cahn':
        driver = allen_cahn_driver
        terminal = allen_cahn_terminal
        exact_y0 = None
    elif name == 'hjb':
        driver = make_hjb_driver(sigma_val)
        terminal = hjb_terminal
        exact_y0 = None  # cole-hopf gives ref but skipping
    elif name == 'bergman':
        driver = make_bergman_driver(0.03, 0.06, sigma_val)
        terminal = bergman_terminal
        exact_y0 = None
    else:
        raise ValueError(f"Unknown benchmark: {name}")

    model = DeepBSDE(dim=dim, T=T, N=N, driver=driver, terminal=terminal,
                     sigma_val=sigma_val, mu_val=mu_val,
                     hidden_dim=hidden_dim, num_layers=num_layers)
    losses, y0_history = train_deep_bsde(
        model, n_epochs=n_epochs, batch_size=batch_size, lr=lr,
        x0_value=x0_value, verbose=True
    )
    return model, losses, y0_history, exact_y0


def dimension_scaling(name, dims, T=1.0, N=20, sigma_val=1.0, n_epochs=500,
                      batch_size=256, lr=1e-3, hidden_dim=64, x0_value=0.0):
    # run benchmark across dims, report Y_0 + rel error
    results = []
    for d in dims:
        print(f"\n--- {name}, d={d} ---")
        model, losses, y0_hist, exact = run_benchmark(
            name, dim=d, T=T, N=N, sigma_val=sigma_val,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            hidden_dim=hidden_dim, x0_value=x0_value
        )
        y0_final = model.y0.item()
        rel_err = None
        if exact is not None and abs(exact) > 1e-12:
            rel_err = abs(y0_final - exact) / abs(exact)
        results.append({
            'dim': d, 'y0': y0_final, 'exact': exact,
            'rel_error': rel_err, 'losses': losses
        })
    return results


def ablation_study(name='black_scholes', dim=10, T=1.0, N=20, sigma_val=1.0,
                   n_epochs=300, batch_size=256, x0_value=0.0):
    # sweep hidden_dim, num_layers, lr
    configs = {
        'hidden_dim': [16, 32, 64, 128],
        'num_layers': [1, 2, 3],
        'lr': [5e-4, 1e-3, 2e-3, 5e-3],
    }

    results = {}

    print("\n=== Ablation: hidden_dim ===")
    hd_results = []
    for hd in configs['hidden_dim']:
        print(f"\n  hidden_dim={hd}")
        model, losses, _, exact = run_benchmark(
            name, dim=dim, T=T, N=N, sigma_val=sigma_val,
            n_epochs=n_epochs, batch_size=batch_size, lr=1e-3,
            hidden_dim=hd, num_layers=2, x0_value=x0_value
        )
        y0 = model.y0.item()
        rel_err = abs(y0 - exact) / abs(exact) if exact and abs(exact) > 1e-12 else None
        hd_results.append({'value': hd, 'y0': y0, 'rel_error': rel_err,
                           'final_loss': losses[-1]})
    results['hidden_dim'] = hd_results

    print("\n=== Ablation: num_layers ===")
    nl_results = []
    for nl in configs['num_layers']:
        print(f"\n  num_layers={nl}")
        model, losses, _, exact = run_benchmark(
            name, dim=dim, T=T, N=N, sigma_val=sigma_val,
            n_epochs=n_epochs, batch_size=batch_size, lr=1e-3,
            hidden_dim=64, num_layers=nl, x0_value=x0_value
        )
        y0 = model.y0.item()
        rel_err = abs(y0 - exact) / abs(exact) if exact and abs(exact) > 1e-12 else None
        nl_results.append({'value': nl, 'y0': y0, 'rel_error': rel_err,
                           'final_loss': losses[-1]})
    results['num_layers'] = nl_results

    print("\n=== Ablation: learning rate ===")
    lr_results = []
    for lr_val in configs['lr']:
        print(f"\n  lr={lr_val}")
        model, losses, _, exact = run_benchmark(
            name, dim=dim, T=T, N=N, sigma_val=sigma_val,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr_val,
            hidden_dim=64, num_layers=2, x0_value=x0_value
        )
        y0 = model.y0.item()
        rel_err = abs(y0 - exact) / abs(exact) if exact and abs(exact) > 1e-12 else None
        lr_results.append({'value': lr_val, 'y0': y0, 'rel_error': rel_err,
                           'final_loss': losses[-1]})
    results['lr'] = lr_results

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("Deep BSDE Solver -- Benchmark Tests")
    print("=" * 60)

    # quick test BS d=10
    model, losses, y0_hist, exact = run_benchmark(
        'black_scholes', dim=10, T=1.0, N=20, sigma_val=1.0,
        n_epochs=500, batch_size=256, lr=5e-3, x0_value=0.0
    )
    print(f"\nBlack-Scholes d=10: Y_0={model.y0.item():.4f}, exact={exact:.4f}")
    print(f"Relative error: {abs(model.y0.item() - exact) / abs(exact):.4f}")
