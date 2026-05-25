import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

from models import (LSTMClassifier, TCNClassifier, TransformerClassifier,
                    create_sequences)


def prepare_data(features, labels, seq_len=50, test_size=0.2,
                 val_size=0.15, seed=42):
    # temporal split -- don't shuffle, respect time ordering
    T = len(features)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    feat_tensor = torch.FloatTensor(features_scaled)
    label_tensor = torch.LongTensor(labels)
    X, y = create_sequences(feat_tensor, label_tensor, seq_len=seq_len)

    N = len(X)

    train_end = int(N * (1 - test_size - val_size))
    val_end = int(N * (1 - test_size))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    batch_size = min(64, train_end)
    # this broke with batch_size > 64 on the full dataset, kept it at 64

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size
    )

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler,
        'n_features': features.shape[1],
        'seq_len': seq_len,
        'n_classes': len(np.unique(labels))
    }


def train_model(model, data, n_epochs=30, lr=1e-3, patience=7, device='cpu'):
    model = model.to(device)
    # tried sgd but adam works fine here
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'epoch_times': []
    }

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in data['train_loader']:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in data['val_loader']:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        epoch_time = time.time() - t0

        train_loss /= max(train_total, 1)
        val_loss /= max(val_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)

        scheduler.step(val_loss)

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, train_acc={train_acc:.3f}, "
                  f"val_acc={val_acc:.3f}, time={epoch_time:.2f}s")

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def evaluate_model(model, data, device='cpu'):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for X_batch, y_batch in data['test_loader']:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(y_batch)
            all_logits.append(logits.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    logits = torch.cat(all_logits).numpy()

    accuracy = (preds == labels).mean()

    n_classes = data['n_classes']
    per_class = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class[c] = {
                'precision': (preds[preds == c] == c).sum() / max((preds == c).sum(), 1),
                'recall': (preds[mask] == c).sum() / mask.sum(),
                'support': int(mask.sum())
            }

    return {
        'accuracy': accuracy,
        'per_class': per_class,
        'predictions': preds,
        'labels': labels,
        'logits': logits
    }


def measure_latency(model, input_shape, n_runs=100, device='cpu'):
    model.eval()
    model.to(device)
    x = torch.randn(1, *input_shape).to(device)

    for _ in range(10):
        with torch.no_grad():
            model(x)

    latencies = []
    for _ in range(n_runs):
        t0 = time.time()
        with torch.no_grad():
            model(x)
        latencies.append((time.time() - t0) * 1000)

    latencies = np.array(latencies)
    return {
        'mean_ms': latencies.mean(),
        'std_ms': latencies.std(),
        'min_ms': latencies.min(),
        'max_ms': latencies.max(),
        'p99_ms': np.percentile(latencies, 99)
    }


def train_all_models(features, labels_mid, labels_spread,
                     seq_len=50, n_epochs=30, device='cpu'):
    n_features = features.shape[1]
    results = {}

    print("\n=== Training LSTM for mid-price prediction ===")
    data_mid = prepare_data(features, labels_mid, seq_len=seq_len)
    lstm = LSTMClassifier(input_dim=n_features, hidden_dim=64,
                          num_layers=2, num_classes=3)
    hist_lstm = train_model(lstm, data_mid, n_epochs=n_epochs, device=device)
    eval_lstm = evaluate_model(lstm, data_mid, device=device)
    lat_lstm = measure_latency(lstm, (seq_len, n_features), device=device)
    print(f"  Test accuracy: {eval_lstm['accuracy']:.3f}")
    print(f"  Latency: {lat_lstm['mean_ms']:.2f} +/- {lat_lstm['std_ms']:.2f} ms")

    results['lstm'] = {
        'model': lstm, 'history': hist_lstm,
        'eval': eval_lstm, 'latency': lat_lstm, 'data': data_mid
    }

    print("\n=== Training GRU for mid-price prediction ===")
    gru = LSTMClassifier(input_dim=n_features, hidden_dim=64,
                         num_layers=2, num_classes=3, use_gru=True)
    hist_gru = train_model(gru, data_mid, n_epochs=n_epochs, device=device)
    eval_gru = evaluate_model(gru, data_mid, device=device)
    lat_gru = measure_latency(gru, (seq_len, n_features), device=device)
    print(f"  Test accuracy: {eval_gru['accuracy']:.3f}")
    print(f"  Latency: {lat_gru['mean_ms']:.2f} +/- {lat_gru['std_ms']:.2f} ms")

    results['gru'] = {
        'model': gru, 'history': hist_gru,
        'eval': eval_gru, 'latency': lat_gru, 'data': data_mid
    }

    print("\n=== Training TCN for spread regime classification ===")
    data_spread = prepare_data(features, labels_spread, seq_len=seq_len)
    tcn = TCNClassifier(input_dim=n_features, num_channels=[32, 32, 32],
                        kernel_size=3, num_classes=3)
    hist_tcn = train_model(tcn, data_spread, n_epochs=n_epochs, device=device)
    eval_tcn = evaluate_model(tcn, data_spread, device=device)
    lat_tcn = measure_latency(tcn, (seq_len, n_features), device=device)
    print(f"  Test accuracy: {eval_tcn['accuracy']:.3f}")
    print(f"  Latency: {lat_tcn['mean_ms']:.2f} +/- {lat_tcn['std_ms']:.2f} ms")

    results['tcn'] = {
        'model': tcn, 'history': hist_tcn,
        'eval': eval_tcn, 'latency': lat_tcn, 'data': data_spread
    }

    print("\n=== Training Transformer for mid-price prediction ===")
    transformer = TransformerClassifier(
        input_dim=n_features, d_model=64, nhead=4,
        num_layers=2, dim_feedforward=128, num_classes=3
    )
    hist_tf = train_model(transformer, data_mid, n_epochs=n_epochs, device=device)
    eval_tf = evaluate_model(transformer, data_mid, device=device)
    lat_tf = measure_latency(transformer, (seq_len, n_features), device=device)
    print(f"  Test accuracy: {eval_tf['accuracy']:.3f}")
    print(f"  Latency: {lat_tf['mean_ms']:.2f} +/- {lat_tf['std_ms']:.2f} ms")

    results['transformer'] = {
        'model': transformer, 'history': hist_tf,
        'eval': eval_tf, 'latency': lat_tf, 'data': data_mid
    }

    return results


if __name__ == '__main__':
    from simulate_lob import generate_lob_data
    from features import extract_all_features

    print("Generating LOB data...")
    data = generate_lob_data(T=10.0, dt=0.01, seed=42)
    feats = extract_all_features(data, horizon=10)

    print(f"Features shape: {feats['features'].shape}")
    print(f"Mid-price labels: {np.bincount(feats['midprice_labels'])}")
    print(f"Spread labels: {np.bincount(feats['spread_labels'])}")

    results = train_all_models(
        feats['features'], feats['midprice_labels'], feats['spread_labels'],
        seq_len=50, n_epochs=20
    )

    print("\n=== Summary ===")
    for name, res in results.items():
        print(f"{name}: acc={res['eval']['accuracy']:.3f}, "
              f"latency={res['latency']['mean_ms']:.2f}ms")
