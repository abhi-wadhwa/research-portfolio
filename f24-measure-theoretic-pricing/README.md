# Notes on Measure Theory and Fine Properties of Functions

**Author:** Abhi Wadhwa  
**Semester:** Fall 2024

## Overview

Study notes on Evans & Gariepy, *Measure Theory and Fine Properties of Functions* (1992 CRC edition). Six chapters covering:

1. **General Measure Theory** -- Outer measures, Caratheodory's construction, Lebesgue integral, MCT/DCT, Fubini--Tonelli, covering theorems (Vitali, Besicovitch), differentiation of Radon measures, Lebesgue points, Riesz representation, weak convergence of measures.
2. **Hausdorff Measure** -- Hausdorff dimension, isodiametric inequality via Steiner symmetrization, densities, the identity $\mathcal{H}^n = \mathcal{L}^n$.
3. **Area and Coarea Formulas** -- Lipschitz functions, Rademacher's theorem, Jacobians, the area formula (change of variables for Lipschitz maps), the coarea formula (nonlinear Fubini).
4. **Sobolev Functions** -- Weak derivatives, approximation (Meyers--Serrin), traces, extensions, Gagliardo--Nirenberg--Sobolev and Morrey inequalities, Rellich--Kondrachov compactness, capacity, quasicontinuity.
5. **BV Functions and Sets of Finite Perimeter** -- Variation, structure theorem, approximation and compactness, coarea formula for BV, isoperimetric inequality, reduced boundary, Gauss--Green theorem, pointwise properties.
6. **Differentiability and Approximation by C^1 Functions** -- $L^p$ differentiability, approximate differentiability, Aleksandrov's theorem for convex functions, Whitney extension, the $C^1$ approximation theorem for Sobolev functions.

These are working notes, not a polished treatment. Proofs are sometimes sketched or deferred to the book. The informal voice and TODO items are intentional.

## Repository structure

```
.
├── README.md
├── paper/
│   ├── main.tex                # Wrapper document (article class)
│   ├── ch1.tex                 # Chapter 1: General Measure Theory
│   ├── ch2.tex                 # Chapter 2: Hausdorff Measure
│   ├── ch3.tex                 # Chapter 3: Area and Coarea Formulas
│   ├── ch4.tex                 # Chapter 4: Sobolev Functions
│   ├── ch5.tex                 # Chapter 5: BV Functions
│   ├── ch6.tex                 # Chapter 6: Differentiability, C^1 Approx
│   └── references.bib          # Bibliography (not used by notes)
└── code/
    ├── README.md
    ├── requirements.txt
    ├── filtrations.py           # Sigma-algebra / filtration visualizations
    ├── lebesgue_vs_riemann.py   # Lebesgue vs Riemann integration demos
    ├── convergence.py           # Lp and martingale convergence demos
    ├── black_scholes.py         # Risk-neutral pricing / Girsanov demos
    ├── run_all.py               # Generate all 14 figures
    └── figures/                 # Generated PDF plots (gitignored)
```

## Generating figures

```bash
cd code
pip install -r requirements.txt
python run_all.py
```

## Compiling the notes

```bash
cd paper
pdflatex main
pdflatex main   # second pass for cross-references
```

No bibliography pass is needed (the notes don't cite external references).

## Acknowledgments

Thanks to Brian Fan for many helpful discussions while working through this material.
