#!/usr/bin/env python3
# runs everything, generates all 14 figs

import time
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("generating figures...")

start = time.time()

import filtrations
filtrations.run_all()

import lebesgue_vs_riemann
lebesgue_vs_riemann.run_all()

import convergence
convergence.run_all()

import black_scholes
black_scholes.run_all()

elapsed = time.time() - start
print(f"\ndone in {elapsed:.1f}s")
print(f"output: {os.path.join(os.getcwd(), 'figures')}")

figdir = os.path.join(os.getcwd(), "figures")
figs = sorted(os.listdir(figdir))
print(f"\n{len(figs)} figures:")
for f in figs:
    sz = os.path.getsize(os.path.join(figdir, f)) / 1024
    print(f"  {f:45s} ({sz:.0f} KB)")
