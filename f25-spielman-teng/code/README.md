# spielman-teng simulations

monte carlo experiments verifying the sah-sahasrabudhe-sawhney bound on least singular values of random matrices.

## usage

```bash
pip install -r requirements.txt
python generate_all_figures.py
```

## files

- simulate_random_matrices.py — gaussian & rademacher ensembles, edelman density, CDF comparisons
- perturbation_analysis.py — convergence of P/eps ratio, error term decay
- spectral_conditioning.py — condition number scaling, smoothed analysis experiments
- generate_all_figures.py — runs everything
