#!/usr/bin/env python3
# run_configurable_simulation.py
# Run the script with: python run_configurable_simulation.py config_name
# Runs economic simulation based on a specified configuration file

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import time
import os
import argparse
import importlib.util


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run economic simulation with specified configuration"
    )
    parser.add_argument(
        "config_name",
        type=str,
        help="Name of the configuration file (without .py extension)",
    )
    args = parser.parse_args()

    config_name = args.config_name
    config_path = f"config/{config_name}.py"

    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found.")
        sys.exit(1)

    # Dynamically import the specified configuration
    try:
        spec = importlib.util.spec_from_file_location(config_name, config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        print(f"Successfully imported configuration from {config_path}")
    except Exception as e:
        print(f"Error importing configuration file: {e}")
        sys.exit(1)

    # Import model equations
    try:
        from model_equations import (
            nested_ces_production,
            update_capital,
            nested_ces_marginal_products,
            allocate_labor,
            allocate_capital_investment,
        )

        print("Successfully imported model functions.")
    except ImportError:
        print("\n" + "=" * 50)
        print("Error: Could not import from model_equations.py.")
        print("Ensure the file exists in the same directory and has no errors.")
        print("=" * 50 + "\n")
        sys.exit(1)

    print("-" * 40)

    # --- Load Parameters from Config ---
    T_sim = config.T_sim
    L_total = config.L_total
    years = np.arange(T_sim + 1)  # Generate years based on T_sim

    print(f"Simulation Time: {T_sim} years")

    # Generate S-shaped curves using parameters loaded from config
    phi_T_t = config.phi_T_init + (config.phi_T_max - config.phi_T_init) / (
        1 + np.exp(-config.phi_T_k * (years - config.phi_T_t0))
    )
    phi_T_t[0] = config.phi_T_init

    phi_I_t = config.phi_I_init + (config.phi_I_max - config.phi_I_init) / (
        1 + np.exp(-config.phi_I_k * (years - config.phi_I_t0))
    )
    phi_I_t[0] = config.phi_I_init

    print(f"AI Adoption (phi_A_share) initial: T={phi_T_t[0]:.2f}, I={phi_I_t[0]:.2f}")
    print(
        f"AI Adoption (phi_A_share) final:   T={phi_T_t[-1]:.2f}, I={phi_I_t[-1]:.2f}"
    )
    print("-" * 40)

    # Load initial conditions from config
    initial_conditions = config.initial_conditions
    L_T_0 = initial_conditions["L_T_0"]
    L_H_0 = initial_conditions["L_H_0"]
    L_I_0 = initial_conditions["L_I_0"]
    L_U_0 = initial_conditions["L_U_0"]
    K_T_0 = initial_conditions["K_T_0"]
    K_H_0 = initial_conditions["K_H_0"]
    K_I_0 = initial_conditions["K_I_0"]
    A_T_0 = initial_conditions["A_T_0"]
    A_H_0 = initial_conditions["A_H_0"]
    A_I_0 = initial_conditions["A_I_0"]

    # Verify initial labor allocation
    initial_sum = L_T_0 + L_H_0 + L_I_0 + L_U_0
    if not math.isclose(initial_sum, L_total):
        print(
            f"Warning: Initial labor allocation ({initial_sum}) doesn't sum to L_total ({L_total}). Normalizing."
        )
        scale = L_total / initial_sum
        L_T_0 *= scale
        L_H_0 *= scale
        L_I_0 *= scale
        L_U_0 *= scale
    print("Initial labor allocation:")
    print(
        f"  L_T={L_T_0:.2f}, L_H={L_H_0:.2f}, L_I={L_I_0:.2f}, L_U={L_U_0:.2f} (Total={L_total:.2f})"
    )
    print("-" * 40)

    print("Initial Capital Stocks:")
    print(f"  K: T={K_T_0:.2f}, H={K_H_0:.2f}, I={K_I_0:.2f}")
    print(f"  A: T={A_T_0:.2f}, H={A_H_0:.2f}, I={A_I_0:.2f}")
    print("-" * 40)

    # Load economic parameters from config
    economic_params = config.economic_params
    delta_K = economic_params["delta_K"]
    delta_A = economic_params["delta_A"]
    s_K = economic_params["s_K"]
    s_A = economic_params["s_A"]
    capital_sensitivity = economic_params["capital_sensitivity"]

    print("Economic Parameters:")
    print(f"  delta_K={delta_K}, delta_A={delta_A}")
    print(f"  Aggregate Savings Rates: s_K={s_K}, s_A={s_A}")
    print(f"  Capital Allocation Sensitivity: {capital_sensitivity}")
    print("-" * 40)

    # Load production parameters from config
    production_params = config.base_production_params
    params_T = production_params["T"]
    params_H = production_params["H"]
    params_I = production_params["I"]

    print("Production Parameters:")
    print(f"  Sector T: {params_T}")
    print(f"  Sector H: {params_H}")
    print(f"  Sector I: {params_I}")
    print("-" * 40)

    # Load labor mobility parameters from config
    labor_mobility_params = config.labor_mobility_params

    print("Labor Mobility Parameters:")
    for k, v in labor_mobility_params.items():
        print(f"  {k}: {v}")
    print("-" * 40)

    # --- Data Storage ---
    # Initialize numpy arrays
    Y_T, Y_H, Y_I = np.zeros(T_sim + 1), np.zeros(T_sim + 1), np.zeros(T_sim + 1)
    K_T, K_H, K_I = np.zeros(T_sim + 1), np.zeros(T_sim + 1), np.zeros(T_sim + 1)
    A_T, A_H, A_I = np.zeros(T_sim + 1), np.zeros(T_sim + 1), np.zeros(T_sim + 1)
    L_T, L_H, L_I, L_U = (
        np.zeros(T_sim + 1),
        np.zeros(T_sim + 1),
        np.zeros(T_sim + 1),
        np.zeros(T_sim + 1),
    )
    MPK_T, MPA_T, MPL_T = np.zeros(T_sim + 1), np.zeros(T_sim + 1), np.zeros(T_sim + 1)
    MPK_H, MPA_H, MPL_H = np.zeros(T_sim + 1), np.zeros(T_sim + 1), np.zeros(T_sim + 1)
    MPK_I, MPA_I, MPL_I = np.zeros(T_sim + 1), np.zeros(T_sim + 1), np.zeros(T_sim + 1)
    InvK_T, InvK_H, InvK_I = (
        np.zeros(T_sim + 1),
        np.zeros(T_sim + 1),
        np.zeros(T_sim + 1),
    )
    InvA_T, InvA_H, InvA_I = (
        np.zeros(T_sim + 1),
        np.zeros(T_sim + 1),
        np.zeros(T_sim + 1),
    )

    # --- Initialize Year 0 ---
    print("Initializing Year 0...")
    K_T[0], K_H[0], K_I[0] = K_T_0, K_H_0, K_I_0
    A_T[0], A_H[0], A_I[0] = A_T_0, A_H_0, A_I_0
    L_T[0], L_H[0], L_I[0], L_U[0] = L_T_0, L_H_0, L_I_0, L_U_0

    # Calculate initial output and MPs using parameters from config
    Y_T[0], _ = nested_ces_production(
        K_T[0],
        A_T[0],
        L_T[0],
        params_T["alpha"],
        phi_T_t[0],
        params_T["rho_outer"],
        params_T["rho_inner"],
    )
    Y_H[0], _ = nested_ces_production(K_H[0], A_H[0], L_H[0], **params_H)
    Y_I[0], _ = nested_ces_production(
        K_I[0],
        A_I[0],
        L_I[0],
        params_I["alpha"],
        phi_I_t[0],
        params_I["rho_outer"],
        params_I["rho_inner"],
    )

    MPK_T[0], MPA_T[0], MPL_T[0] = nested_ces_marginal_products(
        K_T[0],
        A_T[0],
        L_T[0],
        params_T["alpha"],
        phi_T_t[0],
        params_T["rho_outer"],
        params_T["rho_inner"],
    )
    MPK_H[0], MPA_H[0], MPL_H[0] = nested_ces_marginal_products(
        K_H[0], A_H[0], L_H[0], **params_H
    )
    MPK_I[0], MPA_I[0], MPL_I[0] = nested_ces_marginal_products(
        K_I[0],
        A_I[0],
        L_I[0],
        params_I["alpha"],
        phi_I_t[0],
        params_I["rho_outer"],
        params_I["rho_inner"],
    )

    print(f"  Initial Total Output: {Y_T[0]+Y_H[0]+Y_I[0]:.2f}")
    print(
        f"  Initial MPs (K): MPK_T={MPK_T[0]:.4f}, MPK_H={MPK_H[0]:.4f}, MPK_I={MPK_I[0]:.4f}"
    )
    print(
        f"  Initial MPs (A): MPA_T={MPA_T[0]:.4f}, MPA_H={MPA_H[0]:.4f}, MPA_I={MPA_I[0]:.4f}"
    )
    print(
        f"  Initial MPs (L): MPL_T={MPL_T[0]:.4f}, MPL_H={MPL_H[0]:.4f}, MPL_I={MPL_I[0]:.4f}"
    )
    print("-" * 40)

    # --- Simulation Loop ---
    print("Starting Simulation Loop...")
    start_sim_time = time.time()

    for t in range(T_sim):

        # 1. Determine Investment Allocation based on previous period's state (t)
        InvK_T[t], InvK_H[t], InvK_I[t], InvA_T[t], InvA_H[t], InvA_I[t] = (
            allocate_capital_investment(
                Y_T[t],
                Y_H[t],
                Y_I[t],
                MPK_T[t],
                MPK_H[t],
                MPK_I[t],
                MPA_T[t],
                MPA_H[t],
                MPA_I[t],
                s_K,
                s_A,
                capital_sensitivity,
            )
        )
        InvA_H[t] = 0.0

        # 2. Update Capital Stocks using allocated investment
        K_T[t + 1] = update_capital(K_T[t], InvK_T[t], delta_K)
        K_H[t + 1] = update_capital(K_H[t], InvK_H[t], delta_K)
        K_I[t + 1] = update_capital(K_I[t], InvK_I[t], delta_K)
        A_T[t + 1] = update_capital(A_T[t], InvA_T[t], delta_A)
        A_H[t + 1] = 0.0  # Remains 0
        A_I[t + 1] = update_capital(A_I[t], InvA_I[t], delta_A)

        # 3. Allocate Labor for period t+1
        # Create temporary param dicts including the correct phi_A_share for this timestep (t+1)
        current_params_T_for_alloc = params_T.copy()
        current_params_T_for_alloc["phi_A_share"] = phi_T_t[t + 1]

        current_params_I_for_alloc = params_I.copy()
        current_params_I_for_alloc["phi_A_share"] = phi_I_t[t + 1]

        L_T[t + 1], L_H[t + 1], L_I[t + 1], L_U[t + 1] = allocate_labor(
            current_L_T=L_T[t],
            current_L_H=L_H[t],
            current_L_I=L_I[t],
            current_L_U=L_U[t],
            K_T=K_T[t + 1],
            A_T=A_T[t + 1],
            K_H=K_H[t + 1],
            A_H=A_H[t + 1],
            K_I=K_I[t + 1],
            A_I=A_I[t + 1],
            params_T=current_params_T_for_alloc,
            params_H=params_H,
            params_I=current_params_I_for_alloc,
            L_total=L_total,
            **labor_mobility_params,
        )

        # 4. Calculate Output for period t+1 using the NEW capital AND NEW labor allocation
        current_T_phi = phi_T_t[t + 1]
        current_I_phi = phi_I_t[t + 1]
        Y_T[t + 1], _ = nested_ces_production(
            K_T[t + 1],
            A_T[t + 1],
            L_T[t + 1],
            params_T["alpha"],
            current_T_phi,
            params_T["rho_outer"],
            params_T["rho_inner"],
        )
        Y_H[t + 1], _ = nested_ces_production(
            K_H[t + 1], A_H[t + 1], L_H[t + 1], **params_H
        )
        Y_I[t + 1], _ = nested_ces_production(
            K_I[t + 1],
            A_I[t + 1],
            L_I[t + 1],
            params_I["alpha"],
            current_I_phi,
            params_I["rho_outer"],
            params_I["rho_inner"],
        )

        # 5. Calculate Marginal Products for period t+1 (reflecting the end-of-period state)
        MPK_T[t + 1], MPA_T[t + 1], MPL_T[t + 1] = nested_ces_marginal_products(
            K_T[t + 1],
            A_T[t + 1],
            L_T[t + 1],
            params_T["alpha"],
            current_T_phi,
            params_T["rho_outer"],
            params_T["rho_inner"],
        )
        MPK_H[t + 1], MPA_H[t + 1], MPL_H[t + 1] = nested_ces_marginal_products(
            K_H[t + 1], A_H[t + 1], L_H[t + 1], **params_H
        )
        MPK_I[t + 1], MPA_I[t + 1], MPL_I[t + 1] = nested_ces_marginal_products(
            K_I[t + 1],
            A_I[t + 1],
            L_I[t + 1],
            params_I["alpha"],
            current_I_phi,
            params_I["rho_outer"],
            params_I["rho_inner"],
        )

    end_sim_time = time.time()
    print(f"Simulation Loop Finished in {end_sim_time - start_sim_time:.2f} seconds.")
    print("-" * 40)

    # --- Final Results Summary ---
    total_output_final = Y_T[-1] + Y_H[-1] + Y_I[-1]
    print(f"Final State (Year {T_sim}):")
    print(f"  Total Output = {total_output_final:.2f}")
    print("  Labor Allocation:")
    print(f"    L_T = {L_T[-1]:.2f}")
    print(f"    L_H = {L_H[-1]:.2f}")
    print(f"    L_I = {L_I[-1]:.2f}")
    print(f"    L_U = {L_U[-1]:.2f} (Unemployment Rate: {L_U[-1]/L_total:.1%})")
    print(
        f"    Check Sum = {L_T[-1] + L_H[-1] + L_I[-1] + L_U[-1]:.2f} (vs Total {L_total:.2f})"
    )
    print("  Sector Outputs:")
    print(f"    Y_T = {Y_T[-1]:.2f}")
    print(f"    Y_H = {Y_H[-1]:.2f}")
    print(f"    Y_I = {Y_I[-1]:.2f}")
    print("  Sector Capital (K):")
    print(f"    K_T = {K_T[-1]:.2f}")
    print(f"    K_H = {K_H[-1]:.2f}")
    print(f"    K_I = {K_I[-1]:.2f}")
    print("  Sector Capital (A):")
    print(f"    A_T = {A_T[-1]:.2f}")
    print(f"    A_H = {A_H[-1]:.2f}")
    print(f"    A_I = {A_I[-1]:.2f}")
    print("  Marginal Products (K):")
    print(f"    MPK_T = {MPK_T[-1]:.4f}")
    print(f"    MPK_H = {MPK_H[-1]:.4f}")
    print(f"    MPK_I = {MPK_I[-1]:.4f}")
    print("  Marginal Products (A):")
    print(f"    MPA_T = {MPA_T[-1]:.4f}")
    print(f"    MPA_H = {MPA_H[-1]:.4f}")  # Should be 0
    print(f"    MPA_I = {MPA_I[-1]:.4f}")
    print("  Marginal Products (L) / Wages:")
    print(f"    MPL_T = {MPL_T[-1]:.4f}")
    print(f"    MPL_H = {MPL_H[-1]:.4f}")
    print(f"    MPL_I = {MPL_I[-1]:.4f}")
    print("-" * 40)

    # --- Plotting Results ---
    print("Generating and Saving Plots...")
    plt.style.use("seaborn-v0_8-darkgrid")

    # Create a directory for plots named after the config
    plot_dir = f"../results/plots/plots_{config_name}"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Plot 0: Total Output
    plt.figure(figsize=(6, 3))
    Y_total = Y_T + Y_H + Y_I
    plt.plot(
        years,
        Y_total,
        label="Total Output ($Y$)",
        marker="o",
        markersize=4,
        lw=2,
        color="black",
    )
    plt.title(f"{config_name}: Total Output")
    plt.xlabel("Year")
    plt.ylabel("Total Output")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "plot_total_output.png"))
    plt.close()

    # Plot 1: AI Adoption Path (phi_A_share driver)
    plt.figure(figsize=(6, 3))
    plt.plot(
        years, phi_T_t, label=r"Traditional Sector Potential ($\phi_T$)", marker="."
    )
    plt.plot(
        years, phi_I_t, label=r"Intelligence Sector Potential ($\phi_I$)", marker="."
    )
    plt.title(r"Configurable: S-shaped AI Adoption Potential ($\phi_A$)")
    plt.xlabel("Year")
    plt.ylabel(r"Potential AI Share ($\phi$)")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "plot_phi_adoption.png"))
    plt.close()

    # Plot 2: Sector Outputs
    plt.figure(figsize=(6, 3))
    plt.plot(years, Y_T, label=r"Traditional ($Y_T$)", marker="s", markersize=4)
    plt.plot(years, Y_I, label=r"Intelligence ($Y_I$)", marker="d", markersize=4)
    plt.plot(years, Y_H, label=r"Human ($Y_H$)", marker="^", markersize=4)
    plt.title(f"{config_name}: Sector Outputs")
    plt.xlabel("Year")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "plot_sector_outputs.png"))
    plt.close()

    # Plot 3a: Sector Capital Stocks (K and A)
    plt.figure(figsize=(6, 3))
    plt.plot(years, K_T, label=r"$K_T$", marker="s", markersize=4)
    plt.plot(years, K_I, label=r"$K_I$", marker="d", markersize=4)
    plt.plot(years, K_H, label=r"$K_H$", marker="^", markersize=4)
    plt.title(f"{config_name}: Sector Capital Stocks")
    plt.xlabel("Year")
    plt.ylabel(r"Capital Stock $K$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "plot_capital_stocks_k.png"))
    plt.close()

    # Plot 3b: Sector Capital Stocks (K and A)
    plt.figure(figsize=(6, 3))
    plt.plot(years, A_T, label=r"$A_T$", marker="s", markersize=4)
    plt.plot(years, A_I, label=r"$A_I$", marker="d", markersize=4)
    plt.title(f"{config_name}: Sector Capital Stocks")
    plt.xlabel("Year")
    plt.ylabel(r"Capital Stock $A$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "plot_capital_stocks_a.png"))
    plt.close()

    # Plot 4: Wages (MPL)
    plt.figure(figsize=(6, 3))
    plt.plot(years, MPL_T, label=r"$w_T$", marker="s", markersize=4)
    plt.plot(years, MPL_I, label=r"$w_I$", marker="d", markersize=4)
    plt.plot(years, MPL_H, label=r"$w_H$", marker="^", markersize=4)
    plt.title(f"{config_name}: Wages (MPL)")
    plt.xlabel("Year")
    plt.ylabel("Wage (MPL)")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "plot_wages_mpl.png"))
    plt.close()

    # Plot 5: Labor Allocation
    plt.figure(figsize=(8, 3))
    plt.stackplot(
        years,
        L_T,
        L_H,
        L_I,
        L_U,
        labels=[r"$L_T$", r"$L_H$", r"$L_I$", r"$L_U$"],
        alpha=0.8,
    )
    plt.title(f"{config_name}: Labor Allocation Over Time")
    plt.xlabel("Year")
    plt.ylabel("Labor Units")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ylim(0, L_total)
    plt.xlim(0, T_sim)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(plot_dir, "plot_labor_allocation.png"))
    plt.close()

    # Plot 6: Marginal Products of Capital (MPK and MPA)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # Share Y axis
    axes[0].plot(years, MPK_T, label=r"$MPK_T$", marker="s", markersize=4)
    axes[0].plot(years, MPK_H, label=r"$MPK_H$", marker="^", markersize=4)
    axes[0].plot(years, MPK_I, label=r"$MPK_I$", marker="d", markersize=4)
    axes[0].set_title(r"Marginal Product of K ($MPK$)")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Marginal Product")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_ylim(bottom=0)

    axes[1].plot(years, MPA_T, label=r"$MPA_T$", marker="s", markersize=4)
    axes[1].plot(years, MPA_I, label=r"$MPA_I$", marker="d", markersize=4)
    axes[1].plot(
        years,
        MPA_H,
        label=r"$MPA_H$ (Should be 0)",
        marker="^",
        markersize=4,
        linestyle=":",
    )  # MPA_H is always 0
    axes[1].set_title(r"Marginal Product of A ($MPA$)")
    axes[1].set_xlabel("Year")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim(bottom=0)
    fig.suptitle(f"{config_name}: Marginal Returns to Capital Over Time")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(plot_dir, "plot_mpk_mpa.png"))
    plt.close()

    print(f"Plots saved to directory '{plot_dir}'.")
    print(f"\n{config_name} Simulation Run Complete.")


if __name__ == "__main__":
    main()
