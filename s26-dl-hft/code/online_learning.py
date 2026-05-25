import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from models import LSTMClassifier, create_sequences


class ADWIN:
    # drift detector, see bifet & gavalda 2007
    def __init__(self, delta=0.002, min_window=30):
        self.delta = delta
        self.min_window = min_window
        self.window = []
        self.drift_detected = False

    def add_element(self, value):
        self.window.append(value)
        self.drift_detected = False

        if len(self.window) < 2 * self.min_window:
            return False

        # check all split points -- not efficient but works
        n = len(self.window)
        for i in range(self.min_window, n - self.min_window):
            w0 = self.window[:i]
            w1 = self.window[i:]

            n0, n1 = len(w0), len(w1)
            mu0, mu1 = np.mean(w0), np.mean(w1)

            # hoeffding bound
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = np.sqrt((1.0 / (2.0 * m)) * np.log(4.0 / self.delta))

            if abs(mu0 - mu1) >= epsilon:
                self.window = self.window[i:]
                self.drift_detected = True
                return True

        return False

    def reset(self):
        self.window = []
        self.drift_detected = False


class PageHinkley:
    # monitors cumsum deviations from running mean
    def __init__(self, threshold=50.0, alpha=0.005, min_instances=30):
        self.threshold = threshold
        self.alpha = alpha
        self.min_instances = min_instances
        self.reset()

    def add_element(self, value):
        self.n += 1
        self.sum += value
        self.mean = self.sum / self.n

        self.cumsum += value - self.mean - self.alpha
        self.cumsum_min = min(self.cumsum_min, self.cumsum)

        self.drift_detected = False
        if self.n >= self.min_instances:
            if (self.cumsum - self.cumsum_min) > self.threshold:
                self.drift_detected = True
                return True
        return False

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.mean = 0.0
        self.cumsum = 0.0
        self.cumsum_min = float('inf')
        self.drift_detected = False


def online_learning_experiment(features, labels, regime_labels,
                               seq_len=50, window_size=500,
                               retrain_interval=200, eval_interval=50,
                               n_epochs_retrain=10, input_dim=13,
                               device='cpu', seed=42):
    # online learning w/ ADWIN drift detection + sliding window retrain
    torch.manual_seed(seed)
    np.random.seed(seed)

    T = len(features)
    scaler = StandardScaler()

    # small lstm for online setting
    model = LSTMClassifier(input_dim=input_dim, hidden_dim=32,
                           num_layers=1, num_classes=3)
    model.to(device)

    adwin = ADWIN(delta=0.01, min_window=20)
    # tried page-hinkley too but adwin worked better empirically

    eval_times = []
    accuracies_online = []
    accuracies_static = []
    drift_points = []
    retrain_points = []
    regime_at_eval = []

    # initial training on first window
    start = 0
    end = min(window_size + seq_len, T)
    _train_on_window(model, features, labels, scaler, start, end,
                     seq_len, n_epochs_retrain, device)

    # static model = frozen baseline
    static_model = LSTMClassifier(input_dim=input_dim, hidden_dim=32,
                                  num_layers=1, num_classes=3)
    static_model.load_state_dict(model.state_dict())
    static_model.to(device)

    steps_since_retrain = 0

    for t in range(window_size + seq_len, T - eval_interval, eval_interval):
        eval_end = min(t + eval_interval, T)
        if eval_end - seq_len <= t:
            continue

        acc_online = _evaluate_on_window(
            model, features, labels, scaler, t, eval_end, seq_len, device
        )
        acc_static = _evaluate_on_window(
            static_model, features, labels, scaler, t, eval_end, seq_len, device
        )

        eval_times.append(t)
        accuracies_online.append(acc_online)
        accuracies_static.append(acc_static)
        regime_at_eval.append(regime_labels[t] if t < len(regime_labels) else -1)

        error = 1.0 - acc_online
        drift = adwin.add_element(error)
        if drift:
            drift_points.append(t)

        steps_since_retrain += eval_interval

        if drift or steps_since_retrain >= retrain_interval:
            train_start = max(0, t - window_size)
            train_end = min(t + seq_len, T)
            if train_end - train_start > seq_len + 10:
                _train_on_window(model, features, labels, scaler,
                                 train_start, train_end, seq_len,
                                 n_epochs_retrain, device)
                retrain_points.append(t)
                steps_since_retrain = 0

    return {
        'eval_times': np.array(eval_times),
        'accuracies_online': np.array(accuracies_online),
        'accuracies_static': np.array(accuracies_static),
        'drift_points': np.array(drift_points),
        'retrain_points': np.array(retrain_points),
        'regime_at_eval': np.array(regime_at_eval),
        'regime_labels': regime_labels
    }


def _train_on_window(model, features, labels, scaler,
                     start, end, seq_len, n_epochs, device):
    window_feat = features[start:end]
    window_labels = labels[start:end]

    feat_scaled = scaler.fit_transform(window_feat)

    feat_tensor = torch.FloatTensor(feat_scaled)
    label_tensor = torch.LongTensor(window_labels)

    if len(feat_tensor) <= seq_len:
        return

    X, y = create_sequences(feat_tensor, label_tensor, seq_len=seq_len)

    if len(X) == 0:
        return

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=min(32, len(X)), shuffle=True)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(n_epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()


def _evaluate_on_window(model, features, labels, scaler,
                        start, end, seq_len, device):
    window_feat = features[start:end]
    window_labels = labels[start:end]

    feat_scaled = scaler.transform(window_feat)
    feat_tensor = torch.FloatTensor(feat_scaled)
    label_tensor = torch.LongTensor(window_labels)

    if len(feat_tensor) <= seq_len:
        return 0.5  # return chance if window too small

    X, y = create_sequences(feat_tensor, label_tensor, seq_len=seq_len)

    if len(X) == 0:
        return 0.5

    model.eval()
    with torch.no_grad():
        X = X.to(device)
        logits = model(X)
        preds = logits.argmax(1).cpu()
        acc = (preds == y).float().mean().item()

    return acc


if __name__ == '__main__':
    from simulate_lob import generate_multi_regime_data
    from features import extract_all_features

    print("Generating multi-regime LOB data...")
    data = generate_multi_regime_data(T_per_regime=3.0, seed=42)
    feats = extract_all_features(data, horizon=10)

    print(f"Total snapshots: {len(feats['features'])}")
    print(f"Regime sequence: {data['regime_names']}")

    print("\nRunning online learning experiment...")
    results = online_learning_experiment(
        features=feats['features'],
        labels=feats['midprice_labels'],
        regime_labels=data['regime_labels'],
        seq_len=30,
        window_size=300,
        retrain_interval=150,
        eval_interval=30,
        n_epochs_retrain=5,
        input_dim=feats['n_features'],
        seed=42
    )

    print(f"\nEvaluation points: {len(results['eval_times'])}")
    print(f"Drift detections: {len(results['drift_points'])}")
    print(f"Retraining events: {len(results['retrain_points'])}")
    print(f"Mean online accuracy: {results['accuracies_online'].mean():.3f}")
    print(f"Mean static accuracy: {results['accuracies_static'].mean():.3f}")
