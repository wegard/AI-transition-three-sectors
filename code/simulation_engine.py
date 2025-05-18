# simulation_engine.py
import numpy as np
import math
from model_equations import (
    nested_ces_production,
    update_capital,
    nested_ces_marginal_products,
    allocate_labor,
    allocate_capital_investment,
)

# Define epsilon within this file or import if needed elsewhere
epsilon = 1e-9


def run_single_simulation(
    T_sim,
    L_total,
    initial_conditions,  # Dictionary for L_T_0, K_T_0, A_T_0 etc.
    economic_params,  # Dictionary for delta_K, s_K, capital_sensitivity etc.
    production_params,  # Dictionary containing params_T, params_H, params_I
    labor_mobility_params,  # Dictionary for mobility_factor etc.
    phi_T_t,
    phi_I_t,  # Pre-calculated S-curves for phi_A_share drivers
    verbose=False,  # Optional flag to print progress within the simulation
):
    """
    Runs a single simulation of the three-sector model for T_sim periods.

    Args:
        T_sim (int): Number of simulation years.
        L_total (float): Total labor force.
        initial_conditions (dict): Contains initial values like L_T_0, L_H_0, L_I_0, L_U_0,
                                   K_T_0, K_H_0, K_I_0, A_T_0, A_H_0, A_I_0.
        economic_params (dict): Contains parameters like delta_K, delta_A, s_K, s_A,
                                capital_sensitivity.
        production_params (dict): Contains 'T', 'H', 'I' keys, each holding a dict
                                  of production parameters (alpha, rho_outer, rho_inner).
                                  Note: phi_A_share is handled via phi_T_t, phi_I_t.
        labor_mobility_params (dict): Contains parameters for allocate_labor function.
        phi_T_t (np.array): Time series array for T sector AI potential share driver.
        phi_I_t (np.array): Time series array for I sector AI potential share driver.
        verbose (bool): If True, prints status updates during the simulation.

    Returns:
        dict: A dictionary containing the time series results (numpy arrays) for
              key variables like Y_T, L_H, MPL_I, K_T, A_I, etc.
    """
    years = np.arange(T_sim + 1)

    # --- Extract Parameters ---
    delta_K = economic_params["delta_K"]
    delta_A = economic_params["delta_A"]
    s_K = economic_params["s_K"]
    s_A = economic_params["s_A"]
    capital_sensitivity = economic_params["capital_sensitivity"]

    params_T = production_params["T"]
    params_H = production_params["H"]
    params_I = production_params[
        "I"
    ]  # This will contain the specific rho_inner for this run

    # --- Data Storage ---
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
    if verbose:
        print("Initializing Year 0...")
    L_T[0] = initial_conditions["L_T_0"]
    L_H[0] = initial_conditions["L_H_0"]
    L_I[0] = initial_conditions["L_I_0"]
    L_U[0] = initial_conditions["L_U_0"]
    K_T[0] = initial_conditions["K_T_0"]
    K_H[0] = initial_conditions["K_H_0"]
    K_I[0] = initial_conditions["K_I_0"]
    A_T[0] = initial_conditions["A_T_0"]
    A_H[0] = initial_conditions["A_H_0"]  # Should be 0
    A_I[0] = initial_conditions["A_I_0"]

    # Verify initial labor allocation sums to L_total
    initial_sum = L_T[0] + L_H[0] + L_I[0] + L_U[0]
    if not math.isclose(initial_sum, L_total):
        print(
            f"Warning: Initial labor allocation ({initial_sum}) doesn't sum to L_total ({L_total}). Normalizing."
        )
        scale = L_total / initial_sum
        L_T[0] *= scale
        L_H[0] *= scale
        L_I[0] *= scale
        L_U[0] *= scale

    # Calculate initial output and MPs
    # Note: Using params_T/I without phi_A_share key, as it's passed separately below
    Y_T[0], _ = nested_ces_production(
        K_T[0],
        A_T[0],
        L_T[0],
        params_T["alpha"],
        phi_T_t[0],
        params_T["rho_outer"],
        params_T["rho_inner"],
    )
    Y_H[0], _ = nested_ces_production(
        K_H[0], A_H[0], L_H[0], **params_H
    )  # params_H includes phi_A_share=0
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

    if verbose:
        print(f"  Initial Total Output: {Y_T[0]+Y_H[0]+Y_I[0]:.2f}")

    # --- Simulation Loop ---
    if verbose:
        print("Starting Simulation Loop...")
    for t in range(T_sim):
        # 1. Determine Investment Allocation
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

        # 2. Update Capital Stocks
        K_T[t + 1] = update_capital(K_T[t], InvK_T[t], delta_K)
        K_H[t + 1] = update_capital(K_H[t], InvK_H[t], delta_K)
        K_I[t + 1] = update_capital(K_I[t], InvK_I[t], delta_K)
        A_T[t + 1] = update_capital(A_T[t], InvA_T[t], delta_A)
        A_H[t + 1] = 0.0
        A_I[t + 1] = update_capital(A_I[t], InvA_I[t], delta_A)

        # 3. Allocate Labor for period t+1
        # Create temporary param dicts including the correct phi_A_share for this timestep (t+1)
        # for the MPL calculation inside allocate_labor
        current_params_T_for_alloc = params_T.copy()
        current_params_T_for_alloc["phi_A_share"] = phi_T_t[t + 1]

        current_params_I_for_alloc = params_I.copy()
        current_params_I_for_alloc["phi_A_share"] = phi_I_t[t + 1]

        # Note: params_H already includes phi_A_share=0, so no need to update it.

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
            params_T=current_params_T_for_alloc,  # Pass dict with correct phi
            params_H=params_H,
            params_I=current_params_I_for_alloc,  # Pass dict with correct phi AND rho_inner
            L_total=L_total,
            **labor_mobility_params,
        )

        # 4. Calculate Output for period t+1
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

        # 5. Calculate Marginal Products for period t+1
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

        if verbose and (t + 1) % 5 == 0:
            print(f"  Completed Year {t+1}/{T_sim}")

    if verbose:
        print("Simulation Loop Finished.")

    # --- Package Results ---
    results = {
        "years": years,
        "Y_T": Y_T,
        "Y_H": Y_H,
        "Y_I": Y_I,
        "Y_Total": Y_T + Y_H + Y_I,
        "K_T": K_T,
        "K_H": K_H,
        "K_I": K_I,
        "A_T": A_T,
        "A_H": A_H,
        "A_I": A_I,
        "L_T": L_T,
        "L_H": L_H,
        "L_I": L_I,
        "L_U": L_U,
        "MPK_T": MPK_T,
        "MPK_H": MPK_H,
        "MPK_I": MPK_I,
        "MPA_T": MPA_T,
        "MPA_H": MPA_H,
        "MPA_I": MPA_I,
        "MPL_T": MPL_T,
        "MPL_H": MPL_H,
        "MPL_I": MPL_I,
        "InvK_T": InvK_T,
        "InvK_H": InvK_H,
        "InvK_I": InvK_I,
        "InvA_T": InvA_T,
        "InvA_H": InvA_H,
        "InvA_I": InvA_I,
        "phi_T_t": phi_T_t,
        "phi_I_t": phi_I_t,
    }

    return results
