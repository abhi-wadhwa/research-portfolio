# hft signal detection

hawkes process LOB simulation + RNN/TCN/transformer training for mid-price prediction.

## running

```bash
pip install -r requirements.txt
python generate_figures.py
```

## files

- simulate_lob.py -- multivariate hawkes process, limit order book simulator
- features.py -- feature extraction from LOB snapshots (order imbalance, VWAP, etc.)
- models.py -- RNN, TCN, transformer architectures
- train.py -- training loop with early stopping
- online_learning.py -- online gradient descent with regret tracking
- generate_figures.py -- makes all the paper figures
