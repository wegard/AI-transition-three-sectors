#!/usr/bin/env python3
"""
Test script for reallocation experiments.

This script demonstrates how to use the new run_reallocation_experiment function
to study the effects of reallocating resources (K, L, A) between sectors while
keeping totals constant.
"""

import numpy as np
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_experiments import run_reallocation_experiment


def main():
    print("=" * 60)
    print("Testing Reallocation Experiments")
    print("=" * 60)

    # Example 1: Reallocate capital (K) to the Intelligence sector
    print("\n1. Testing Capital Reallocation to Intelligence Sector")
    print("-" * 50)

    results_K_to_I = run_reallocation_experiment(
        variable_type="K",
        target_sector="I",
        reallocation_fractions=np.array(
            [0.2, 0.3, 0.4]
        ),  # I gets 20%, 30%, 40% of total K
        experiment_name="test_K_to_I",
        do_plots=True,
    )

    if results_K_to_I:
        print(f"✓ Capital reallocation experiment completed successfully!")
        print(
            f"  Number of scenarios tested: {len(results_K_to_I) - 1}"
        )  # -1 for baseline
    else:
        print("✗ Capital reallocation experiment failed!")

    # Example 2: Reallocate labor (L) to the Traditional sector
    print("\n2. Testing Labor Reallocation to Traditional Sector")
    print("-" * 50)

    results_L_to_T = run_reallocation_experiment(
        variable_type="L",
        target_sector="T",
        reallocation_fractions=np.array(
            [0.4, 0.5]
        ),  # T gets 40%, 50% of total employed labor
        experiment_name="test_L_to_T",
        do_plots=True,
    )

    if results_L_to_T:
        print(f"✓ Labor reallocation experiment completed successfully!")
        print(f"  Number of scenarios tested: {len(results_L_to_T) - 1}")
    else:
        print("✗ Labor reallocation experiment failed!")

    # Example 3: Reallocate AI capital (A) to the Traditional sector
    print("\n3. Testing AI Capital Reallocation to Traditional Sector")
    print("-" * 50)

    results_A_to_T = run_reallocation_experiment(
        variable_type="A",
        target_sector="T",
        reallocation_fractions=np.array(
            [0.6, 0.8]
        ),  # T gets 60%, 80% of total AI capital
        experiment_name="test_A_to_T",
        do_plots=True,
    )

    if results_A_to_T:
        print(f"✓ AI capital reallocation experiment completed successfully!")
        print(f"  Number of scenarios tested: {len(results_A_to_T) - 1}")
    else:
        print("✗ AI capital reallocation experiment failed!")

    print("\n" + "=" * 60)
    print("All reallocation experiments completed!")
    print("Check the ../results/experiments/ directory for output plots and data.")
    print("=" * 60)


if __name__ == "__main__":
    main()
