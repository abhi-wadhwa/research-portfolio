import time
import sys
import os

# make sure imports work from this dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    t0 = time.time()

    from simulate_random_matrices import main as run_random
    from perturbation_analysis import main as run_perturbation
    from spectral_conditioning import main as run_conditioning

    run_random()
    run_perturbation()
    run_conditioning()

    elapsed = time.time() - t0
    print(f"All figures generated in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
