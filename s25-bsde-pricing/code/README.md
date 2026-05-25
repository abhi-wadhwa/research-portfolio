# bsde pricing code

implementations of SDE/BSDE solvers, deep BSDE method, CVA calculation, and g-expectation risk measures.

## running

```bash
pip install -r requirements.txt
python generate_figures.py      # basic figures
python generate_new_figures.py  # deep bsde + cva + g-expectation figures
```

## files

- sde_simulation.py -- euler-maruyama, milstein for GBM and heston
- bsde_solver.py -- explicit BSDE solver for linear/nonlinear drivers
- pde_solver.py -- finite difference black-scholes PDE solver
- deep_bsde_solver.py -- han-jentzen-e deep BSDE neural net (pytorch)
- cva_bsde.py -- credit valuation adjustment via BSDEs
- g_expectation.py -- g-expectations as risk measures
- generate_figures.py, generate_new_figures.py -- figure generation scripts
