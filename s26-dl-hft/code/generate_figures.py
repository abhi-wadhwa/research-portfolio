import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulate_lob import generate_lob_data, generate_multi_regime_data
from features import extract_all_features, compute_midprice_labels
from models import (LSTMClassifier, TCNClassifier, TransformerClassifier,
                    create_sequences)
from online_learning import online_learning_experiment

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})
sns.set_style("whitegrid")


def generate_shared_data():
    print("Generating LOB data...")
    data = generate_lob_data(T=20.0, dt=0.01, seed=42)
    feats = extract_all_features(data, horizon=10)

    # recompute labels w/ adaptive threshold so classes are balanced
    # default threshold gives like 90% flat which is useless
    mid = feats['mid_prices']
    horizon = 10
    T = len(mid)
    rets = np.zeros(T)
    for i in range(T - horizon):
        rets[i] = (mid[i + horizon] - mid[i]) / (mid[i] + 1e-10)
    p33 = np.percentile(rets[:T - horizon], 33)
    p66 = np.percentile(rets[:T - horizon], 66)
    labels = np.ones(T, dtype=int)
    labels[rets < p33] = 0
    labels[rets > p66] = 2
    labels[T - horizon:] = 1
    feats['midprice_labels'] = labels

    print(f"  Features: {feats['features'].shape}, Labels: {np.bincount(feats['midprice_labels'])}")
    return data, feats


def prepare_sequences(feats, seq_len=50, label_key='midprice_labels'):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feats['features'])
    X, y = create_sequences(
        torch.FloatTensor(feat_scaled),
        torch.LongTensor(feats[label_key]),
        seq_len=seq_len
    )
    return X, y, scaler


def quick_train(model, X_train, y_train, X_val, y_val, n_epochs=8, lr=1e-3):
    # not production quality, just for figures
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    bs = min(64, len(X_train))

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_loss, correct, total = 0.0, 0, 0
        for i in range(0, len(X_train), bs):
            idx = perm[i:i+bs]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        history['train_loss'].append(epoch_loss / max(total, 1))
        history['train_acc'].append(correct / max(total, 1))

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
            val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    return history


def plot_lob_heatmap(data):
    print("Plotting LOB heatmap...")
    mid = data['mid_prices']
    bid_vol = data['bid_volumes']
    ask_vol = data['ask_volumes']
    ts = data['timestamps']

    step = 5
    idx = np.arange(0, len(ts), step)
    ts_sub = ts[idx]
    n_levels = bid_vol.shape[1]

    all_levels = np.arange(-n_levels, n_levels + 1)
    heatmap = np.zeros((len(all_levels), len(idx)))

    for j, i in enumerate(idx):
        for lev in range(n_levels):
            bid_row = n_levels - 1 - lev
            heatmap[bid_row, j] = bid_vol[i, lev]
            ask_row = n_levels + 1 + lev
            heatmap[ask_row, j] = ask_vol[i, lev]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(ts_sub, all_levels * 0.01 * 100, heatmap,
                       cmap='YlOrRd', shading='auto')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Price Level Offset (bps from mid)')
    ax.set_title('Limit Order Book Depth Heatmap')
    cbar = plt.colorbar(im, ax=ax, label='Volume')

    ax2 = ax.twinx()
    ax2.plot(ts_sub, mid[idx], 'b-', linewidth=0.8, alpha=0.7, label='Mid-price')
    ax2.set_ylabel('Mid-price', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.legend(loc='upper right')

    fig.savefig(os.path.join(FIGURES_DIR, 'lob_heatmap.png'))
    plt.close(fig)
    print("  Saved lob_heatmap.png")


def plot_learning_curves(feats):
    print("Plotting learning curves...")
    seq_len = 50
    X, y, _ = prepare_sequences(feats, seq_len=seq_len)

    n = len(X)
    tr_end = int(n * 0.65)
    va_end = int(n * 0.80)
    X_tr, y_tr = X[:tr_end], y[:tr_end]
    X_va, y_va = X[tr_end:va_end], y[tr_end:va_end]

    input_dim = X.shape[2]
    n_epochs = 10

    models_cfg = {
        'LSTM': LSTMClassifier(input_dim=input_dim, hidden_dim=32, num_layers=1, num_classes=3),
        'GRU': LSTMClassifier(input_dim=input_dim, hidden_dim=32, num_layers=1, num_classes=3, use_gru=True),
        'TCN': TCNClassifier(input_dim=input_dim, num_channels=[16, 16], kernel_size=3, num_classes=3),
        'Transformer': TransformerClassifier(input_dim=input_dim, d_model=32, nhead=4,
                                             num_layers=1, dim_feedforward=64, num_classes=3),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = sns.color_palette("Set1", 4)

    histories = {}
    for (name, model), color in zip(models_cfg.items(), colors):
        print(f"  Training {name}...")
        hist = quick_train(model, X_tr, y_tr, X_va, y_va, n_epochs=n_epochs)
        histories[name] = hist

        epochs = range(1, n_epochs + 1)
        axes[0].plot(epochs, hist['train_loss'], '-', color=color, label=f'{name} train')
        axes[0].plot(epochs, hist['val_loss'], '--', color=color, label=f'{name} val')
        axes[1].plot(epochs, hist['train_acc'], '-', color=color, label=f'{name} train')
        axes[1].plot(epochs, hist['val_acc'], '--', color=color, label=f'{name} val')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend(fontsize=7, ncol=2)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'learning_curves.png'))
    plt.close(fig)
    print("  Saved learning_curves.png")
    return histories


def plot_accuracy_vs_latency(feats):
    print("Plotting accuracy vs latency...")
    seq_len = 50
    X, y, _ = prepare_sequences(feats, seq_len=seq_len)

    n = len(X)
    tr_end = int(n * 0.65)
    va_end = int(n * 0.80)
    X_tr, y_tr = X[:tr_end], y[:tr_end]
    X_va, y_va = X[tr_end:va_end], y[tr_end:va_end]
    X_te, y_te = X[va_end:], y[va_end:]

    input_dim = X.shape[2]

    configs = [
        ('LSTM-S', LSTMClassifier(input_dim=input_dim, hidden_dim=16, num_layers=1, num_classes=3)),
        ('LSTM-L', LSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2, num_classes=3)),
        ('GRU-S', LSTMClassifier(input_dim=input_dim, hidden_dim=16, num_layers=1, num_classes=3, use_gru=True)),
        ('GRU-L', LSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2, num_classes=3, use_gru=True)),
        ('TCN-S', TCNClassifier(input_dim=input_dim, num_channels=[16, 16], kernel_size=3, num_classes=3)),
        ('TCN-L', TCNClassifier(input_dim=input_dim, num_channels=[32, 32, 32], kernel_size=3, num_classes=3)),
        ('Transf-S', TransformerClassifier(input_dim=input_dim, d_model=16, nhead=2,
                                           num_layers=1, dim_feedforward=32, num_classes=3)),
        ('Transf-L', TransformerClassifier(input_dim=input_dim, d_model=64, nhead=4,
                                           num_layers=2, dim_feedforward=128, num_classes=3)),
    ]

    names, accs, latencies, param_counts = [], [], [], []
    markers = {'LSTM': 'o', 'GRU': 's', 'TCN': '^', 'Transf': 'D'}
    arch_colors = {'LSTM': 'C0', 'GRU': 'C1', 'TCN': 'C2', 'Transf': 'C3'}

    for name, model in configs:
        print(f"  {name}...")
        quick_train(model, X_tr, y_tr, X_va, y_va, n_epochs=8)
        model.eval()

        with torch.no_grad():
            preds = model(X_te).argmax(1)
            acc = (preds == y_te).float().mean().item()

        x_single = torch.randn(1, seq_len, input_dim)
        for _ in range(10):
            with torch.no_grad():
                model(x_single)
        times = []
        for _ in range(50):
            t0 = time.time()
            with torch.no_grad():
                model(x_single)
            times.append((time.time() - t0) * 1000)
        lat = np.mean(times)

        n_params = sum(p.numel() for p in model.parameters())
        names.append(name)
        accs.append(acc)
        latencies.append(lat)
        param_counts.append(n_params)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, name in enumerate(names):
        arch = name.split('-')[0]
        ax.scatter(latencies[i], accs[i], s=param_counts[i] / 30 + 40,
                   marker=markers[arch], color=arch_colors[arch],
                   edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(name, (latencies[i], accs[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    for arch in markers:
        ax.scatter([], [], marker=markers[arch], color=arch_colors[arch],
                   edgecolors='black', linewidth=0.5, s=80, label=arch)
    ax.legend(title='Architecture', loc='lower right')

    ax.set_xlabel('Inference Latency (ms)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Accuracy vs. Inference Latency (bubble size = parameter count)')

    fig.savefig(os.path.join(FIGURES_DIR, 'accuracy_vs_latency.png'))
    plt.close(fig)
    print("  Saved accuracy_vs_latency.png")


def plot_generalization_gap(feats):
    print("Plotting generalization gap...")
    seq_len = 50
    X, y, _ = prepare_sequences(feats, seq_len=seq_len)

    n = len(X)
    tr_end = int(n * 0.65)
    va_end = int(n * 0.80)
    X_tr, y_tr = X[:tr_end], y[:tr_end]
    X_te, y_te = X[va_end:], y[va_end:]

    input_dim = X.shape[2]

    hidden_dims = [8, 16, 32, 64, 128]

    results = {arch: {'train': [], 'test': []} for arch in ['LSTM', 'GRU', 'TCN', 'Transformer']}

    for hdim in hidden_dims:
        print(f"  hid_dim={hdim}...")
        models_dict = {
            'LSTM': LSTMClassifier(input_dim=input_dim, hidden_dim=hdim, num_layers=1, num_classes=3),
            'GRU': LSTMClassifier(input_dim=input_dim, hidden_dim=hdim, num_layers=1, num_classes=3, use_gru=True),
            'TCN': TCNClassifier(input_dim=input_dim, num_channels=[hdim, hdim], kernel_size=3, num_classes=3),
            'Transformer': TransformerClassifier(input_dim=input_dim, d_model=max(hdim, 4),
                                                  nhead=min(4, max(hdim, 4)),
                                                  num_layers=1, dim_feedforward=hdim*2, num_classes=3),
        }
        for arch_name, model in models_dict.items():
            quick_train(model, X_tr, y_tr, X_te, y_te, n_epochs=8)
            model.eval()
            with torch.no_grad():
                train_acc = (model(X_tr).argmax(1) == y_tr).float().mean().item()
                test_acc = (model(X_te).argmax(1) == y_te).float().mean().item()
            results[arch_name]['train'].append(train_acc)
            results[arch_name]['test'].append(test_acc)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = sns.color_palette("Set1", 4)
    arch_names = list(results.keys())

    for i, arch_name in enumerate(arch_names):
        axes[0].plot(hidden_dims, results[arch_name]['train'], 'o-', color=colors[i],
                     label=f'{arch_name} train')
        axes[0].plot(hidden_dims, results[arch_name]['test'], 's--', color=colors[i],
                     label=f'{arch_name} test')

    axes[0].set_xlabel('Hidden Dimension (Model Capacity)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Train vs. Test Accuracy')
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_xscale('log', base=2)

    for i, arch_name in enumerate(arch_names):
        gap = np.array(results[arch_name]['train']) - np.array(results[arch_name]['test'])
        axes[1].plot(hidden_dims, gap, 'o-', color=colors[i], label=arch_name)

    axes[1].set_xlabel('Hidden Dimension (Model Capacity)')
    axes[1].set_ylabel('Generalization Gap (Train - Test Acc)')
    axes[1].set_title('Generalization Gap vs. Model Capacity')
    axes[1].legend()
    axes[1].set_xscale('log', base=2)
    axes[1].axhline(y=0, color='gray', linestyle=':', linewidth=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'generalization_gap.png'))
    plt.close(fig)
    print("  Saved generalization_gap.png")


def plot_attention_maps(feats):
    print("Plotting attention maps...")
    seq_len = 50
    X, y, _ = prepare_sequences(feats, seq_len=seq_len)

    n = len(X)
    tr_end = int(n * 0.65)
    X_tr, y_tr = X[:tr_end], y[:tr_end]
    X_va, y_va = X[tr_end:int(n*0.8)], y[tr_end:int(n*0.8)]

    input_dim = X.shape[2]
    model = TransformerClassifier(input_dim=input_dim, d_model=32, nhead=4,
                                  num_layers=1, dim_feedforward=64, num_classes=3)
    quick_train(model, X_tr, y_tr, X_va, y_va, n_epochs=8)
    model.eval()

    sample_idx = [0, n // 4, n // 2]
    fig, axes = plt.subplots(len(sample_idx), 4, figsize=(14, 3 * len(sample_idx)))

    class_names = ['Down', 'Flat', 'Up']

    for row, si in enumerate(sample_idx):
        x_sample = X[si:si+1]
        with torch.no_grad():
            logits, attn = model(x_sample, return_attention=True)
        pred = logits.argmax(1).item()
        true_label = y[si].item()
        attn_np = attn[0].numpy()

        for head in range(4):
            ax = axes[row, head] if len(sample_idx) > 1 else axes[head]
            im = ax.imshow(attn_np[head], cmap='viridis', aspect='auto',
                          vmin=0, vmax=attn_np.max())
            if row == 0:
                ax.set_title(f'Head {head+1}', fontsize=10)
            if head == 0:
                ax.set_ylabel(f'Sample {row+1}\nTrue={class_names[true_label]}\nPred={class_names[pred]}',
                             fontsize=8)
            ax.set_xlabel('Key position' if row == len(sample_idx) - 1 else '')
            if head > 0:
                ax.set_yticks([])

    plt.suptitle('Transformer Attention Weights over LOB Sequence', y=1.02, fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'attention_maps.png'))
    plt.close(fig)
    print("  Saved attention_maps.png")


def plot_rademacher_complexity(feats):
    # empirical rademacher via random label memorization, see zhang et al 2017
    print("Plotting Rademacher complexity...")
    seq_len = 50
    X, y, _ = prepare_sequences(feats, seq_len=seq_len)
    input_dim = X.shape[2]

    max_available = len(X)
    sample_sizes = [s for s in [50, 100, 200, 500, 1000] if s <= max_available]
    n_trials = 3

    architectures = {
        'LSTM': lambda: LSTMClassifier(input_dim=input_dim, hidden_dim=32, num_layers=1, num_classes=3),
        'GRU': lambda: LSTMClassifier(input_dim=input_dim, hidden_dim=32, num_layers=1, num_classes=3, use_gru=True),
        'TCN': lambda: TCNClassifier(input_dim=input_dim, num_channels=[16, 16], kernel_size=3, num_classes=3),
        'Transformer': lambda: TransformerClassifier(input_dim=input_dim, d_model=32, nhead=4,
                                                      num_layers=1, dim_feedforward=64, num_classes=3),
    }

    results = {name: {'mean': [], 'std': []} for name in architectures}

    for n_samples in sample_sizes:
        print(f"  n={n_samples}...")
        X_sub = X[:n_samples]

        for arch_name, model_fn in architectures.items():
            trial_complexities = []
            for trial in range(n_trials):
                random_labels = torch.randint(0, 3, (n_samples,))

                model = model_fn()
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                criterion = nn.CrossEntropyLoss()

                model.train()
                actual_n = len(X_sub)
                random_labels_use = random_labels[:actual_n]
                bs = min(64, actual_n)
                for epoch in range(15):
                    perm = torch.randperm(actual_n)
                    for i in range(0, actual_n, bs):
                        idx = perm[i:i+bs]
                        xb = X_sub[idx]
                        yb = random_labels_use[idx]
                        optimizer.zero_grad()
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    preds = model(X_sub).argmax(1)
                    rand_acc = (preds == random_labels_use).float().mean().item()

                # normalize: subtract chance, scale to [0,1]
                complexity = max(rand_acc - 1.0/3.0, 0.0) / (2.0/3.0)
                trial_complexities.append(complexity)

            results[arch_name]['mean'].append(np.mean(trial_complexities))
            results[arch_name]['std'].append(np.std(trial_complexities))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = sns.color_palette("Set1", 4)

    for i, (arch_name, res) in enumerate(results.items()):
        means = np.array(res['mean'])
        stds = np.array(res['std'])
        ax.plot(sample_sizes, means, 'o-', color=colors[i], label=arch_name)
        ax.fill_between(sample_sizes, means - stds, means + stds,
                        alpha=0.15, color=colors[i])

    ns = np.array(sample_sizes, dtype=float)
    theoretical = 0.6 / np.sqrt(ns / ns[0])
    ax.plot(sample_sizes, theoretical, 'k--', linewidth=1.5, alpha=0.6,
            label=r'$\mathcal{O}(1/\sqrt{n})$ bound')

    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Empirical Rademacher Complexity (normalized)')
    ax.set_title('Rademacher Complexity vs. Sample Size')
    ax.legend()
    ax.set_xscale('log', base=2)

    fig.savefig(os.path.join(FIGURES_DIR, 'rademacher_complexity.png'))
    plt.close(fig)
    print("  Saved rademacher_complexity.png")


def plot_ablation_results(feats):
    print("Plotting ablation results...")
    seq_len = 50
    X, y, _ = prepare_sequences(feats, seq_len=seq_len)

    n = len(X)
    tr_end = int(n * 0.65)
    va_end = int(n * 0.80)
    X_tr, y_tr = X[:tr_end], y[:tr_end]
    X_va, y_va = X[tr_end:va_end], y[tr_end:va_end]
    X_te, y_te = X[va_end:], y[va_end:]

    input_dim = X.shape[2]

    # lstm depth x width
    depths = [1, 2, 3]
    widths = [16, 32, 64]
    lstm_grid = np.zeros((len(depths), len(widths)))

    print("  LSTM depth x width...")
    for di, d in enumerate(depths):
        for wi, w in enumerate(widths):
            model = LSTMClassifier(input_dim=input_dim, hidden_dim=w,
                                   num_layers=d, num_classes=3)
            quick_train(model, X_tr, y_tr, X_va, y_va, n_epochs=8)
            model.eval()
            with torch.no_grad():
                acc = (model(X_te).argmax(1) == y_te).float().mean().item()
            lstm_grid[di, wi] = acc

    # transformer d_model x nhead
    d_models = [16, 32, 64]
    n_heads = [1, 2, 4]
    tf_grid = np.zeros((len(d_models), len(n_heads)))

    print("  Transformer d_model x nhead...")
    for di, dm in enumerate(d_models):
        for ni, nh in enumerate(n_heads):
            if dm % nh != 0:
                tf_grid[di, ni] = np.nan
                continue
            model = TransformerClassifier(input_dim=input_dim, d_model=dm, nhead=nh,
                                          num_layers=1, dim_feedforward=dm*2, num_classes=3)
            quick_train(model, X_tr, y_tr, X_va, y_va, n_epochs=8)
            model.eval()
            with torch.no_grad():
                acc = (model(X_te).argmax(1) == y_te).float().mean().item()
            tf_grid[di, ni] = acc

    # tcn depth x kernel
    tcn_depths = [1, 2, 3]
    tcn_kernels = [2, 3, 5]
    tcn_grid = np.zeros((len(tcn_depths), len(tcn_kernels)))

    print("  TCN depth x kernel_size...")
    for di, d in enumerate(tcn_depths):
        for ki, k in enumerate(tcn_kernels):
            channels = [16] * d
            model = TCNClassifier(input_dim=input_dim, num_channels=channels,
                                  kernel_size=k, num_classes=3)
            quick_train(model, X_tr, y_tr, X_va, y_va, n_epochs=8)
            model.eval()
            with torch.no_grad():
                acc = (model(X_te).argmax(1) == y_te).float().mean().item()
            tcn_grid[di, ki] = acc

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    sns.heatmap(lstm_grid, ax=axes[0], annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=widths, yticklabels=depths, vmin=0.2, vmax=0.7)
    axes[0].set_xlabel('Hidden Dim')
    axes[0].set_ylabel('Num Layers')
    axes[0].set_title('LSTM: Depth vs. Width')

    mask = np.isnan(tf_grid)
    sns.heatmap(tf_grid, ax=axes[1], annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=n_heads, yticklabels=d_models, mask=mask, vmin=0.2, vmax=0.7)
    axes[1].set_xlabel('Num Heads')
    axes[1].set_ylabel('d_model')
    axes[1].set_title('Transformer: d_model vs. Heads')

    sns.heatmap(tcn_grid, ax=axes[2], annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=tcn_kernels, yticklabels=tcn_depths, vmin=0.2, vmax=0.7)
    axes[2].set_xlabel('Kernel Size')
    axes[2].set_ylabel('Num Layers')
    axes[2].set_title('TCN: Depth vs. Kernel Size')

    plt.suptitle('Architecture Ablation Studies (Test Accuracy)', y=1.02, fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'ablation_results.png'))
    plt.close(fig)
    print("  Saved ablation_results.png")


def plot_concept_drift():
    print("Plotting concept drift...")
    print("  Generating multi-regime data...")
    mr_data = generate_multi_regime_data(T_per_regime=3.0, dt=0.01, seed=42)
    feats = extract_all_features(mr_data, horizon=10)

    print("  Running online learning experiment...")
    results = online_learning_experiment(
        features=feats['features'],
        labels=feats['midprice_labels'],
        regime_labels=mr_data['regime_labels'],
        seq_len=30,
        window_size=300,
        retrain_interval=150,
        eval_interval=30,
        n_epochs_retrain=5,
        input_dim=feats['n_features'],
        seed=42
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})

    eval_times = results['eval_times']
    total_snapshots = len(mr_data['regime_labels'])
    regime_names = ['Normal', 'Volatile', 'Illiquid', 'Normal']

    axes[0].plot(eval_times, results['accuracies_online'], 'b-', linewidth=1.5,
                 label='Online (ADWIN + retrain)', alpha=0.9)
    axes[0].plot(eval_times, results['accuracies_static'], 'r--', linewidth=1.5,
                 label='Static (no retrain)', alpha=0.9)

    if len(results['drift_points']) > 0:
        for dp in results['drift_points']:
            axes[0].axvline(x=dp, color='orange', linestyle=':', alpha=0.6, linewidth=1)
        axes[0].axvline(x=results['drift_points'][0], color='orange', linestyle=':',
                        alpha=0.6, linewidth=1, label='Drift detected')

    if len(results['retrain_points']) > 0:
        axes[0].scatter(results['retrain_points'],
                        [results['accuracies_online'][np.argmin(np.abs(eval_times - rp))]
                         for rp in results['retrain_points']],
                        marker='v', color='green', s=40, zorder=5, label='Retrain event')

    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Online Learning with Concept Drift Detection')
    axes[0].legend(loc='lower left', fontsize=9)
    axes[0].set_ylim([0, 1.05])

    regime_boundaries = [0]
    snapshots_per_regime = total_snapshots // 4
    for i in range(1, 4):
        regime_boundaries.append(i * snapshots_per_regime)
    regime_boundaries.append(total_snapshots)

    regime_colors = ['#d4edda', '#f8d7da', '#fff3cd', '#d4edda']
    for i in range(4):
        for ax in axes:
            ax.axvspan(regime_boundaries[i], regime_boundaries[i+1],
                       alpha=0.2, color=regime_colors[i])
        mid_x = (regime_boundaries[i] + regime_boundaries[i+1]) / 2
        axes[0].text(mid_x, 1.0, regime_names[i], ha='center', va='bottom',
                     fontsize=9, style='italic')

    regime_ts = np.arange(total_snapshots)
    axes[1].fill_between(regime_ts, 0, 1,
                          where=mr_data['regime_labels'] == 0, alpha=0.5,
                          color='green', label='Normal')
    axes[1].fill_between(regime_ts, 0, 1,
                          where=mr_data['regime_labels'] == 1, alpha=0.5,
                          color='red', label='Volatile')
    axes[1].fill_between(regime_ts, 0, 1,
                          where=mr_data['regime_labels'] == 2, alpha=0.5,
                          color='gold', label='Illiquid')
    axes[1].fill_between(regime_ts, 0, 1,
                          where=mr_data['regime_labels'] == 3, alpha=0.5,
                          color='green')
    axes[1].set_xlabel('Time Step (snapshot index)')
    axes[1].set_ylabel('Regime')
    axes[1].set_yticks([])
    axes[1].legend(loc='center right', fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'concept_drift.png'))
    plt.close(fig)
    print("  Saved concept_drift.png")


if __name__ == '__main__':
    print("=" * 60)
    print("Generating all figures for DL-HFT paper")
    print("=" * 60)

    data, feats = generate_shared_data()

    plot_lob_heatmap(data)
    plot_learning_curves(feats)
    plot_accuracy_vs_latency(feats)
    plot_generalization_gap(feats)
    plot_attention_maps(feats)
    plot_rademacher_complexity(feats)
    plot_ablation_results(feats)
    plot_concept_drift()

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 60)
