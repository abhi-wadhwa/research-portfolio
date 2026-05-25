# measure theory figures

python scripts that generate the figures for my evans & gariepy notes.

## running

```bash
pip install -r requirements.txt
python run_all.py
```

generates all 14 figures in figures/. takes about 10 seconds.

## files

- filtrations.py — binomial tree paths, sigma-algebra partitions, conditional expectation
- lebesgue_vs_riemann.py — lebesgue vs riemann integration comparison, fat cantor sets
- convergence.py — modes of convergence (L^p, a.e., in measure), martingale convergence
- black_scholes.py — option pricing surface, girsanov/radon-nikodym, replicating strategies
- run_all.py — runs everything
