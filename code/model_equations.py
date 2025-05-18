import math
import random  # Potentially useful for stochastic extensions, not currently used

# --- Constants ---
epsilon = 1e-9  # Small positive number to avoid division by zero or log(0) issues


##############################
# === Production Function ===
##############################
def nested_ces_production(K, A, L, alpha, phi_A_share, rho_outer, rho_inner):
    """
    Calculates output using a two-level nested Constant Elasticity of Substitution (CES) function.

    Structure:
    - Outer Nest: Combines traditional capital (K) with an aggregate of AI capital and Labor (H).
      Y = [ alpha * K^rho_outer + (1 - alpha) * H^rho_outer ]^(1 / rho_outer)
    - Inner Nest: Combines AI capital (A) and Labor (L) to form the aggregate H.
      H = [ phi_A_share * A^rho_inner + (1 - phi_A_share) * L^rho_inner ]^(1 / rho_inner)

    Parameters:
    - K (float): Traditional capital input (e.g., machinery, buildings). Must be non-negative.
    - A (float): AI capital input (e.g., software, models, compute). Must be non-negative.
    - L (float): Labor input (e.g., hours worked, number of workers). Must be non-negative.
    - alpha (float): Share parameter for K in the outer CES nest.
                     Role: Determines the relative importance/contribution of K vs. H in production.
                     Range: [0.0, 1.0].
                     Effect: Higher alpha increases sensitivity of output Y to changes in K.
                             Lower alpha increases sensitivity to H (and thus A, L, phi_A_share).
    - phi_A_share (float): Share parameter for A within the inner A-L nest (H).
                           Role: Determines the relative importance/contribution of A vs. L in forming H.
                           Range: [0.0, 1.0].
                           Effect: Higher phi_A_share increases sensitivity of H (and Y) to changes in A.
                                   A value of 0 means only L contributes to H (H=L).
                                   A value of 1 means only A contributes to H (H=A).
    - rho_outer (float): Substitution parameter for the outer K-H nest.
                         Role: Governs the ease of substitution between K and H.
                         Related to elasticity sigma_outer = 1 / (1 - rho_outer).
                         Range: (-inf, 1.0]. rho_outer=1 implies perfect substitutes. rho_outer=0 implies Cobb-Douglas. rho_outer<0 implies complements.
                         Effect: Higher rho_outer (closer to 1) means K and H are better substitutes.
                                 Lower rho_outer (more negative) means K and H are stronger complements.
    - rho_inner (float): Substitution parameter for the inner A-L nest.
                         Role: Governs the ease of substitution between A and L. CRITICAL for automation dynamics.
                         Related to elasticity sigma_inner = 1 / (1 - rho_inner).
                         Range: (-inf, 1.0]. rho_inner=1 implies perfect substitutes. rho_inner=0 implies Cobb-Douglas. rho_inner<0 implies complements.
                         Effect: Higher rho_inner (closer to 1) means A and L are better substitutes (AI easily replaces labor).
                                 Lower rho_inner (more negative) means A and L are stronger complements (AI tools boost labor productivity).

    Returns:
    - tuple: (Y, H)
        - Y (float): Final output.
        - H (float): Value of the inner A-L aggregate (useful intermediate value).
    """
    # Ensure inputs are non-negative
    K_eff = max(K, 0.0)
    A_eff = max(A, 0.0)
    L_eff = max(L, 0.0)

    # --- Parameter Validation (Optional but good practice) ---
    # Basic clamping to avoid issues if slightly outside bounds due to float precision
    alpha = max(0.0, min(1.0, alpha))
    phi_A_share = max(0.0, min(1.0, phi_A_share))
    # rho parameters typically <= 1, but allow flexibility here. CES is undefined for rho > 1 if base becomes negative.

    # --- Calculate Inner Aggregate H(A, L) ---
    H = 0.0
    weight_A = phi_A_share  # Effective weight for AI capital A in the H nest
    weight_L = 1.0 - phi_A_share  # Effective weight for Labor L in the H nest

    # --- Handle Special Cases for rho_inner ---

    # Case 1: rho_inner approaches 0 (Cobb-Douglas Limit: sigma_inner = 1)
    if abs(rho_inner) < epsilon:
        # H = A^weight_A * L^weight_L
        # If a required input (with weight > 0) is zero, output H is zero.
        if (weight_A > epsilon and A_eff <= epsilon) or (
            weight_L > epsilon and L_eff <= epsilon
        ):
            H = 0.0
        else:
            try:
                # Calculate terms only if inputs and weights are positive to avoid log(0)
                term_A = (
                    math.log(A_eff) * weight_A
                    if weight_A > epsilon and A_eff > epsilon
                    else 0.0
                )
                term_L = (
                    math.log(L_eff) * weight_L
                    if weight_L > epsilon and L_eff > epsilon
                    else 0.0
                )
                # Combine in log space and exponentiate
                H = math.exp(term_A + term_L)
            except (ValueError, OverflowError):  # Handle potential math errors
                H = 0.0

    # Case 2: rho_inner approaches 1 (Linear/Perfect Substitutes Limit: sigma_inner = inf)
    elif abs(rho_inner - 1.0) < epsilon:
        # H = weight_A * A + weight_L * L
        H = weight_A * A_eff + weight_L * L_eff

    # Case 3: General CES for inner nest (rho_inner != 0 and rho_inner != 1)
    else:
        # Calculate individual terms: weight * input^rho
        term_A = (
            weight_A * (A_eff**rho_inner)
            if weight_A > epsilon and A_eff > epsilon
            else 0.0
        )
        term_L = (
            weight_L * (L_eff**rho_inner)
            if weight_L > epsilon and L_eff > epsilon
            else 0.0
        )

        # Sum the terms: base_H = (weight_A * A^rho_inner + weight_L * L^rho_inner)
        # Handle negative rho (complements): If a required input is zero, the combined base is zero.
        if rho_inner < 0 and (
            (weight_A > epsilon and A_eff <= epsilon)
            or (weight_L > epsilon and L_eff <= epsilon)
        ):
            base_H = 0.0
        else:
            base_H = term_A + term_L

        # Calculate H = base_H ^ (1 / rho_inner)
        if base_H < epsilon:  # If the base sum is effectively zero
            H = 0.0
        else:
            try:
                # Protect against fractional exponent of negative base (undefined in real numbers)
                # This shouldn't happen if inputs K,A,L >= 0 and rho <= 1, but added as safeguard.
                if base_H < 0 and (1.0 / rho_inner) % 1 != 0:
                    H = 0.0  # Or raise an error, depending on desired behavior
                else:
                    # Calculate the final aggregate H
                    H = math.pow(base_H, 1.0 / rho_inner)
            except (ValueError, OverflowError):  # Handle potential math errors
                H = 0.0

    # Ensure H is non-negative after calculations
    H = max(H, 0.0)

    # --- Calculate Outer Production Y(K, H) ---
    Y = 0.0
    weight_K = alpha  # Effective weight for Traditional Capital K in the Y nest
    weight_H = 1.0 - alpha  # Effective weight for the H aggregate in the Y nest

    # --- Handle Special Cases for rho_outer ---

    # Case 1: rho_outer approaches 0 (Cobb-Douglas Limit: sigma_outer = 1)
    if abs(rho_outer) < epsilon:
        # Y = K^weight_K * H^weight_H
        # If a required input (with weight > 0) is zero, output Y is zero.
        if (weight_K > epsilon and K_eff <= epsilon) or (
            weight_H > epsilon and H <= epsilon
        ):
            Y = 0.0
        else:
            try:
                # Calculate terms only if inputs and weights are positive
                term_K = (
                    math.log(K_eff) * weight_K
                    if weight_K > epsilon and K_eff > epsilon
                    else 0.0
                )
                term_H = (
                    math.log(H) * weight_H
                    if weight_H > epsilon and H > epsilon
                    else 0.0
                )
                # Combine in log space and exponentiate
                Y = math.exp(term_K + term_H)
            except (ValueError, OverflowError):
                Y = 0.0

    # Case 2: rho_outer approaches 1 (Linear/Perfect Substitutes Limit: sigma_outer = inf)
    elif abs(rho_outer - 1.0) < epsilon:
        # Y = weight_K * K + weight_H * H
        Y = weight_K * K_eff + weight_H * H

    # Case 3: General CES for outer nest (rho_outer != 0 and rho_outer != 1)
    else:
        # Calculate individual terms: weight * input^rho
        term_K = (
            weight_K * (K_eff**rho_outer)
            if weight_K > epsilon and K_eff > epsilon
            else 0.0
        )
        term_H = (
            weight_H * (H**rho_outer) if weight_H > epsilon and H > epsilon else 0.0
        )

        # Sum the terms: base_Y = (weight_K * K^rho_outer + weight_H * H^rho_outer)
        # Handle negative rho (complements): If a required input is zero, the combined base is zero.
        if rho_outer < 0 and (
            (weight_K > epsilon and K_eff <= epsilon)
            or (weight_H > epsilon and H <= epsilon)
        ):
            base_Y = 0.0
        else:
            base_Y = term_K + term_H

        # Calculate Y = base_Y ^ (1 / rho_outer)
        if base_Y < epsilon:  # If the base sum is effectively zero
            Y = 0.0
        else:
            try:
                # Protect against fractional exponent of negative base
                if base_Y < 0 and (1.0 / rho_outer) % 1 != 0:
                    Y = 0.0
                else:
                    # Calculate the final output Y
                    Y = math.pow(base_Y, 1.0 / rho_outer)
            except (ValueError, OverflowError):
                Y = 0.0

    # Return the final output Y and the intermediate aggregate H, ensuring non-negativity
    return max(Y, 0.0), H


##############################
# === Capital Accumulation ===
##############################
def update_capital(capital_old, investment, delta_per_step):
    """
    Updates the capital stock for one time step using the standard capital accumulation equation.

    Equation: K_new = (1 - delta) * K_old + Investment

    Parameters:
    - capital_old (float): Capital stock at the beginning of the period.
    - investment (float): Gross investment during the period. Must be non-negative.
    - delta_per_step (float): Depreciation rate over the duration of one time step.
                              Range: [0.0, 1.0]. Represents fraction of capital depreciating.

    Returns:
    - float: Capital stock at the end of the period (beginning of next), ensured non-negative.
    """
    # Ensure investment is non-negative
    inv_eff = max(0.0, investment)
    # Calculate new capital stock: depreciated old stock + new investment
    capital_new = (1.0 - delta_per_step) * capital_old + inv_eff
    # Ensure capital stock does not fall below zero
    return max(0.0, capital_new)


# === Marginal Products ===


def nested_ces_marginal_products(K, A, L, alpha, phi_A_share, rho_outer, rho_inner):
    """
    Calculates the marginal products of K, A, and L for the nested CES production function.

    Uses the chain rule for MPL and MPA:
    - MPK = dY/dK
    - MPA = dY/dH * dH/dA
    - MPL = dY/dH * dH/dL

    Parameters:
    - Same parameters as nested_ces_production.

    Returns:
    - tuple: (MPK, MPA, MPL)
        - MPK (float): Marginal product of traditional capital K.
        - MPA (float): Marginal product of AI capital A.
        - MPL (float): Marginal product of Labor L (often interpreted as the real wage).
    """
    # Ensure non-negative inputs for calculation consistency
    K_eff = max(K, 0.0)
    A_eff = max(A, 0.0)
    L_eff = max(L, 0.0)

    # Recalculate Y and H using the effective non-negative inputs
    # This ensures consistency and provides the necessary intermediate values Y and H
    Y, H = nested_ces_production(
        K_eff, A_eff, L_eff, alpha, phi_A_share, rho_outer, rho_inner
    )

    # If output Y is zero, all marginal products must be zero.
    if Y <= epsilon:
        return (0.0, 0.0, 0.0)

    # Initialize marginal products to zero
    MPK, MPA, MPL = 0.0, 0.0, 0.0

    # --- Calculate MPK = dY/dK ---
    weight_K = alpha
    # MPK is positive only if K has a positive weight, K is positive, and Y is positive
    if weight_K > epsilon and K_eff > epsilon:
        try:
            if abs(rho_outer) < epsilon:  # Cobb-Douglas Outer Limit
                # dY/dK = alpha * (Y / K)
                MPK = weight_K * Y / K_eff
            elif abs(rho_outer - 1.0) < epsilon:  # Linear Outer Limit
                # dY/dK = alpha (constant)
                MPK = weight_K
            else:  # General CES Outer
                # dY/dK = alpha * (Y / K)^(1 - rho_outer)  (Using alternative form)
                # Or equivalently: alpha * K^(rho_outer-1) * Y^(1-rho_outer)
                MPK = (
                    weight_K
                    * math.pow(K_eff, rho_outer - 1.0)
                    * math.pow(Y, 1.0 - rho_outer)
                )
        except (ValueError, OverflowError, ZeroDivisionError):
            MPK = 0.0  # Default to zero if calculation fails

    # --- Calculate dY/dH (Intermediate term for chain rule) ---
    dY_dH = 0.0
    weight_H = 1.0 - alpha
    # dY/dH is positive only if H has a positive weight and H is positive (and Y > 0)
    if weight_H > epsilon and H > epsilon:
        try:
            if abs(rho_outer) < epsilon:  # Cobb-Douglas Outer Limit
                # dY/dH = (1 - alpha) * (Y / H)
                dY_dH = weight_H * Y / H
            elif abs(rho_outer - 1.0) < epsilon:  # Linear Outer Limit
                # dY/dH = (1 - alpha) (constant)
                dY_dH = weight_H
            else:  # General CES Outer
                # dY/dH = (1 - alpha) * (Y / H)^(1 - rho_outer)
                # Or equivalently: (1 - alpha) * H^(rho_outer-1) * Y^(1-rho_outer)
                dY_dH = (
                    weight_H
                    * math.pow(H, rho_outer - 1.0)
                    * math.pow(Y, 1.0 - rho_outer)
                )
        except (ValueError, OverflowError, ZeroDivisionError):
            dY_dH = 0.0

    # --- Calculate dH/dL (Intermediate term for chain rule) ---
    dH_dL = 0.0
    weight_L = 1.0 - phi_A_share
    # dH/dL is positive only if L has a positive weight, L is positive, and H is positive
    # Also need L_eff > epsilon to avoid division by zero in CD case
    if weight_L > epsilon and L_eff > epsilon and H > epsilon:
        try:
            if abs(rho_inner) < epsilon:  # Cobb-Douglas Inner Limit
                # dH/dL = (1 - phi) * (H / L)
                dH_dL = weight_L * H / L_eff
            elif abs(rho_inner - 1.0) < epsilon:  # Linear Inner Limit
                # dH/dL = (1 - phi) (constant)
                dH_dL = weight_L
            else:  # General CES Inner
                # dH/dL = (1 - phi) * (H / L)^(1 - rho_inner)
                # Or equivalently: (1 - phi) * L^(rho_inner-1) * H^(1-rho_inner)
                dH_dL = (
                    weight_L
                    * math.pow(L_eff, rho_inner - 1.0)
                    * math.pow(H, 1.0 - rho_inner)
                )
        except (ValueError, OverflowError, ZeroDivisionError):
            dH_dL = 0.0

    # --- Calculate dH/dA (Intermediate term for chain rule) ---
    dH_dA = 0.0
    weight_A = phi_A_share
    # dH/dA is positive only if A has a positive weight, A is positive, and H is positive
    # Also need A_eff > epsilon to avoid division by zero in CD case
    if weight_A > epsilon and A_eff > epsilon and H > epsilon:
        try:
            if abs(rho_inner) < epsilon:  # Cobb-Douglas Inner Limit
                # dH/dA = phi * (H / A)
                dH_dA = weight_A * H / A_eff
            elif abs(rho_inner - 1.0) < epsilon:  # Linear Inner Limit
                # dH/dA = phi (constant)
                dH_dA = weight_A
            else:  # General CES Inner
                # dH/dA = phi * (H / A)^(1 - rho_inner)
                # Or equivalently: phi * A^(rho_inner-1) * H^(1-rho_inner)
                dH_dA = (
                    weight_A
                    * math.pow(A_eff, rho_inner - 1.0)
                    * math.pow(H, 1.0 - rho_inner)
                )
        except (ValueError, OverflowError, ZeroDivisionError):
            dH_dA = 0.0

    # --- Combine derivatives using Chain Rule ---
    # MPL = dY/dH * dH/dL
    MPL = dY_dH * dH_dL
    # MPA = dY/dH * dH/dA
    MPA = dY_dH * dH_dA

    # --- Final Checks and Cleanup ---
    # Ensure marginal products are non-negative and finite
    MPK = 0.0 if not math.isfinite(MPK) or MPK < 0 else MPK
    MPA = 0.0 if not math.isfinite(MPA) or MPA < 0 else MPA
    MPL = 0.0 if not math.isfinite(MPL) or MPL < 0 else MPL

    # Explicitly set MPL to 0 if Labor input is zero
    if L_eff <= epsilon:
        MPL = 0.0
    # Explicitly set MPA to 0 if AI input is zero OR if AI share parameter is zero
    if A_eff <= epsilon or weight_A <= epsilon:
        MPA = 0.0

    return (MPK, MPA, MPL)


##########################
# === Labor Allocation ===
##########################
def allocate_labor(
    current_L_T,
    current_L_H,
    current_L_I,
    current_L_U,  # Current labor distribution
    K_T,
    A_T,
    K_H,
    A_H,
    K_I,
    A_I,  # Capital stocks (used for MPL calculation)
    params_T,
    params_H,
    params_I,  # Production parameters (INCLUDING time-varying phi)
    L_total,  # Total labor force (assumed constant)
    mobility_factor,
    non_movable_fraction,  # Labor market friction parameters
    job_finding_rate,
    job_separation_rate,  # Unemployment dynamics parameters
    wage_sensitivity,  # Sensitivity of flows to wage differences
):
    """
    Allocates the total labor force (L_total) across three sectors (T, H, I)
    and an unemployment pool (U) for the *next* time step, based on current
    conditions and parameters governing frictions and responsiveness.

    Mechanism Overview:
    1. Calculate current wages (MPL) in each sector based on provided capital & labor.
    2. Apply job separations: A fraction of employed workers moves to unemployment.
    3. Hire from unemployment: Available unemployed workers move to sectors based on relative wages.
    4. Reallocate employed workers: Movable workers shift between sectors towards
       higher wages, limited by mobility factor and non-movable fraction.
    5. Ensure conservation of total labor, adjusting if necessary.

    Args:
        current_L_T, _H, _I, _U (float): Labor stocks in each category at the start of the period.
        K_T, A_T, K_H, A_H, K_I, A_I (float): Capital stocks (used to calculate MPLs).
        params_T, params_H, params_I (dict): Dictionaries containing the production function parameters
                                             (alpha, phi_A_share, rho_outer, rho_inner) for each sector.
                                             Crucially, phi_A_share for T and I should reflect the *current* period's value.
        L_total (float): The total size of the labor force (constant).
        mobility_factor (float): Fraction [0,1] of desired inter-sector flow that occurs per step.
        non_movable_fraction (float): Fraction [0,1] of employed workers unable/unwilling to switch sectors.
        job_finding_rate (float): Fraction [0,1] of unemployed finding a job per step.
        job_separation_rate (float): Fraction [0,1] of employed becoming unemployed per step (base churn).
        wage_sensitivity (float): Positive exponent controlling responsiveness to wage diffs (higher=more responsive).

    Returns:
        tuple: (new_L_T, new_L_H, new_L_I, new_L_U) - Labor distribution for the start of the next period.
    """
    # --- 1. Calculate Current Wages (MPL) ---
    # Note: MPL calculation needs the full parameter set for each sector, including the
    # potentially time-varying phi_A_share for the current period.
    # Use max(current_L, epsilon) to avoid issues if a sector has zero labor momentarily.
    _, _, MPL_T = nested_ces_marginal_products(
        K_T, A_T, max(current_L_T, epsilon), **params_T
    )
    _, _, MPL_H = nested_ces_marginal_products(
        K_H, A_H, max(current_L_H, epsilon), **params_H
    )
    _, _, MPL_I = nested_ces_marginal_products(
        K_I, A_I, max(current_L_I, epsilon), **params_I
    )

    # Wages are driven by MPL but cannot be negative
    w_T = max(0.0, MPL_T)
    w_H = max(0.0, MPL_H)
    w_I = max(0.0, MPL_I)

    # --- Initialize labor stocks for modification ---
    L_T, L_H, L_I, L_U = current_L_T, current_L_H, current_L_I, current_L_U

    # --- 2. Job Separations (Flow into Unemployment) ---
    # A fraction of employed workers separates into unemployment each period.
    sep_T = L_T * job_separation_rate
    sep_H = L_H * job_separation_rate
    sep_I = L_I * job_separation_rate

    # Update stocks: Decrease employed, increase unemployed
    L_T = max(0.0, L_T - sep_T)  # Ensure non-negative
    L_H = max(0.0, L_H - sep_H)
    L_I = max(0.0, L_I - sep_I)
    L_U += sep_T + sep_H + sep_I

    # --- 3. Hiring from Unemployment (Flow out of Unemployment) ---
    # Only a fraction of the unemployed actively find jobs in a given step.
    L_U_available = L_U * job_finding_rate
    hired_total = 0.0
    hired_T, hired_H, hired_I = 0.0, 0.0, 0.0

    # Proceed only if there are unemployed workers available to be hired.
    if L_U_available > epsilon:
        # Calculate hiring attractiveness based on wages (powered by sensitivity)
        # Using max(0, w) ensures only non-negative wages attract hiring.
        weight_T = max(0.0, w_T) ** wage_sensitivity
        weight_H = max(0.0, w_H) ** wage_sensitivity
        weight_I = max(0.0, w_I) ** wage_sensitivity
        total_weight = weight_T + weight_H + weight_I

        # Distribute available unemployed proportionally to weights if total weight > 0
        if total_weight > epsilon:
            share_T = weight_T / total_weight
            share_H = weight_H / total_weight
            share_I = weight_I / total_weight

            hired_T = L_U_available * share_T
            hired_H = L_U_available * share_H
            hired_I = L_U_available * share_I
            hired_total = hired_T + hired_H + hired_I

            # --- Sanity check/Rescaling (usually not needed with this method) ---
            # Ensure we don't hire more than available due to floating point issues.
            if hired_total > L_U_available * (1 + epsilon):
                scale = L_U_available / hired_total
                hired_T *= scale
                hired_H *= scale
                hired_I *= scale
                hired_total = L_U_available

            # Update stocks: Increase employed, decrease unemployed
            L_T += hired_T
            L_H += hired_H
            L_I += hired_I
            L_U -= hired_total
        # If total_weight is zero (all wages are zero), no hiring occurs from unemployment.

    # Ensure unemployment remains non-negative
    L_U = max(0.0, L_U)

    # --- 4. Sector-to-Sector Reallocation (Among Currently Employed) ---
    # This step models employed workers switching jobs between sectors towards higher wages.
    L_employed_before_realloc = L_T + L_H + L_I
    if (
        L_employed_before_realloc > epsilon
    ):  # Only proceed if there are employed workers

        # --- Calculate Target Labor Shares using Softmax Logic ---
        # This determines the 'ideal' distribution based on relative wages.
        # Subtracting max wage term improves numerical stability for exp().
        max_wage_term = max(
            w_T, w_H, w_I, 0.0
        )  # Ensure non-negative base for comparison
        try:
            exp_T = math.exp(wage_sensitivity * (w_T - max_wage_term))
            exp_H = math.exp(wage_sensitivity * (w_H - max_wage_term))
            exp_I = math.exp(wage_sensitivity * (w_I - max_wage_term))
        except (
            OverflowError
        ):  # Handle potential overflow if wage diffs * sensitivity is huge
            # Fallback: Assign share only to the max wage sector(s) if exp fails
            max_w = max(w_T, w_H, w_I)
            is_max_T = 1.0 if math.isclose(w_T, max_w) else 0.0
            is_max_H = 1.0 if math.isclose(w_H, max_w) else 0.0
            is_max_I = 1.0 if math.isclose(w_I, max_w) else 0.0
            num_max = is_max_T + is_max_H + is_max_I
            exp_T = is_max_T / num_max if num_max > 0 else 1.0 / 3.0
            exp_H = is_max_H / num_max if num_max > 0 else 1.0 / 3.0
            exp_I = is_max_I / num_max if num_max > 0 else 1.0 / 3.0

        total_exp_wage = exp_T + exp_H + exp_I

        # Calculate target shares if possible, otherwise maintain current distribution
        if total_exp_wage > epsilon:
            target_share_T = exp_T / total_exp_wage
            target_share_H = exp_H / total_exp_wage
            target_share_I = exp_I / total_exp_wage
        else:
            # Keep current shares if all exp() values are zero (e.g., all wages are -inf)
            target_share_T = L_T / L_employed_before_realloc
            target_share_H = L_H / L_employed_before_realloc
            target_share_I = L_I / L_employed_before_realloc

        # Calculate target labor levels based on these shares
        target_L_T = L_employed_before_realloc * target_share_T
        target_L_H = L_employed_before_realloc * target_share_H
        target_L_I = L_employed_before_realloc * target_share_I

        # --- Determine Desired Changes and Flows ---
        # Positive desired change means sector wants workers, negative means it has excess.
        desired_change_T = target_L_T - L_T
        desired_change_H = target_L_H - L_H
        desired_change_I = target_L_I - L_I

        # Potential outflow from sectors with excess workers (negative desired change)
        potential_outflow_T = max(0.0, -desired_change_T)
        potential_outflow_H = max(0.0, -desired_change_H)
        potential_outflow_I = max(0.0, -desired_change_I)

        # Limit outflow by the fraction of workers who are actually movable
        movable_L_T = L_T * (1.0 - non_movable_fraction)
        movable_L_H = L_H * (1.0 - non_movable_fraction)
        movable_L_I = L_I * (1.0 - non_movable_fraction)

        actual_outflow_T = min(potential_outflow_T, movable_L_T)
        actual_outflow_H = min(potential_outflow_H, movable_L_H)
        actual_outflow_I = min(potential_outflow_I, movable_L_I)

        # Total pool of workers leaving shrinking sectors
        total_outflow = actual_outflow_T + actual_outflow_H + actual_outflow_I

        # --- Distribute Outflow Pool to Growing Sectors ---
        # Weight inflow desire by the positive desired change
        inflow_weight_T = max(0.0, desired_change_T)
        inflow_weight_H = max(0.0, desired_change_H)
        inflow_weight_I = max(0.0, desired_change_I)
        total_inflow_weight = inflow_weight_T + inflow_weight_H + inflow_weight_I

        actual_inflow_T, actual_inflow_H, actual_inflow_I = 0.0, 0.0, 0.0
        # Distribute proportionally if there's outflow and desire for inflow
        if total_outflow > epsilon and total_inflow_weight > epsilon:
            actual_inflow_T = total_outflow * (inflow_weight_T / total_inflow_weight)
            actual_inflow_H = total_outflow * (inflow_weight_H / total_inflow_weight)
            actual_inflow_I = total_outflow * (inflow_weight_I / total_inflow_weight)
        # If no sector wants inflow (all shrinking or stable), the outflow doesn't go anywhere inter-sector.

        # --- Apply Mobility Factor ---
        # Only a fraction of the potential flow actually happens due to friction.
        final_outflow_T = mobility_factor * actual_outflow_T
        final_outflow_H = mobility_factor * actual_outflow_H
        final_outflow_I = mobility_factor * actual_outflow_I

        final_inflow_T = mobility_factor * actual_inflow_T
        final_inflow_H = mobility_factor * actual_inflow_H
        final_inflow_I = mobility_factor * actual_inflow_I

        # Update labor stocks based on the final, friction-limited flows
        L_T = L_T - final_outflow_T + final_inflow_T
        L_H = L_H - final_outflow_H + final_inflow_H
        L_I = L_I - final_outflow_I + final_inflow_I

    # Ensure non-negativity after reallocation
    L_T = max(0.0, L_T)
    L_H = max(0.0, L_H)
    L_I = max(0.0, L_I)

    # --- 5. Conservation Check and Normalization ---
    # Ensure the sum L_T + L_H + L_I + L_U equals L_total.
    current_total_L = L_T + L_H + L_I + L_U
    if not math.isclose(current_total_L, L_total, rel_tol=1e-7):
        # If discrepancy exists, adjust employed labor proportionally, keeping L_U fixed.
        L_employed_final = L_T + L_H + L_I
        # Target employed labor is total minus the calculated unemployment
        target_L_employed = max(0.0, L_total - L_U)

        if L_employed_final > epsilon and target_L_employed >= 0:
            # Calculate scaling factor
            scale_factor = target_L_employed / L_employed_final
            # Apply scaling factor
            L_T *= scale_factor
            L_H *= scale_factor
            L_I *= scale_factor
        elif target_L_employed <= epsilon:
            # If target employed is zero (all unemployment), set employed to zero
            L_T, L_H, L_I = 0.0, 0.0, 0.0
            L_U = L_total  # Ensure L_U covers the total labor force
        # If L_employed_final was zero but target > 0, it's an edge case; keep zeros.

    # Final assignment ensuring non-negativity
    new_L_T = max(0.0, L_T)
    new_L_H = max(0.0, L_H)
    new_L_I = max(0.0, L_I)
    # Recalculate L_U as the residual to guarantee conservation exactly
    new_L_U = max(0.0, L_total - (new_L_T + new_L_H + new_L_I))

    # --- Final Safety Check (should ideally not be needed after recalc) ---
    # This ensures the sum is exactly L_total, adjusting L_U primarily.
    final_sum = new_L_T + new_L_H + new_L_I + new_L_U
    if not math.isclose(final_sum, L_total, rel_tol=1e-7):
        # If still off, force L_U to be the exact residual.
        # print(f"CRITICAL WARNING: Labor conservation failed! Sum: {final_sum}, Total: {L_total}") # Reduce noise
        new_L_U = L_total - (new_L_T + new_L_H + new_L_I)
        new_L_U = max(0.0, new_L_U)  # Ensure L_U is not negative
        # If L_U became negative, it means T+H+I > L_total. Need to clip them.
        if new_L_U < 0:
            # print("WARN: Clipping required as T+H+I > L_total") # Reduce noise
            overflow = (new_L_T + new_L_H + new_L_I) - L_total
            total_employed = new_L_T + new_L_H + new_L_I
            if total_employed > epsilon:  # Avoid division by zero
                # Reduce proportionally
                new_L_T -= overflow * (new_L_T / total_employed)
                new_L_H -= overflow * (new_L_H / total_employed)
                new_L_I -= overflow * (new_L_I / total_employed)
            # Ensure non-negativity after clipping
            new_L_T = max(0.0, new_L_T)
            new_L_H = max(0.0, new_L_H)
            new_L_I = max(0.0, new_L_I)
            new_L_U = 0.0  # Set unemployment to zero in this overflow case

    return new_L_T, new_L_H, new_L_I, new_L_U


#######################################
# === Capital Investment Allocation ===
#######################################
def allocate_capital_investment(
    Y_T,
    Y_H,
    Y_I,  # Current total output levels for each sector
    MPK_T,
    MPK_H,
    MPK_I,  # Current marginal products of traditional capital (K)
    MPA_T,
    MPA_H,
    MPA_I,  # Current marginal products of AI capital (A)
    s_K,
    s_A,  # Aggregate savings rates out of TOTAL output for K and A
    capital_sensitivity,  # Sensitivity parameter for allocation based on MPs
    epsilon=1e-9,  # Small number for numerical stability
):
    """
    Allocates total economy-wide investment funds for Traditional Capital (K) and
    AI Capital (A) across the three sectors (T, H, I) based on their respective
    marginal products (MPK and MPA). Sectors with higher marginal returns attract
    a larger share of the investment pool.

    Mechanism:
    1. Calculate the total pools of investment funds available for K and A,
       based on aggregate output (Y_T + Y_H + Y_I) and aggregate savings rates (s_K, s_A).
    2. Calculate allocation weights for each sector based on their current MPK (for K funds)
       and MPA (for A funds). Weights are typically MP^(sensitivity). Only positive MPs attract investment.
    3. Normalize these weights to get allocation shares for each sector and capital type.
    4. Distribute the total investment pools according to these calculated shares.

    Args:
        Y_T, Y_H, Y_I (float): Output levels of the three sectors in the current period.
        MPK_T, MPK_H, MPK_I (float): Marginal products of K in each sector in the current period.
        MPA_T, MPA_H, MPA_I (float): Marginal products of A in each sector in the current period.
                                     (Note: MPA_H is expected to be 0 based on model setup).
        s_K (float): Fraction [0,1] of *total* economy output saved/invested in K.
        s_A (float): Fraction [0,1] of *total* economy output saved/invested in A.
        capital_sensitivity (float): Positive exponent controlling responsiveness of investment
                                     to MP differences (higher = more responsive).
        epsilon (float): Small number for stability checks.

    Returns:
        tuple: (Inv_K_T, Inv_K_H, Inv_K_I, Inv_A_T, Inv_A_H, Inv_A_I)
               The gross investment amounts allocated to each sector for K and A for the next period.
    """

    # --- 1. Calculate Total Investment Pools ---
    # Total output generated in the economy this period
    total_output = Y_T + Y_H + Y_I
    # Total funds available for K investment = aggregate savings rate for K * total output
    total_investment_K = max(0.0, s_K * total_output)
    # Total funds available for A investment = aggregate savings rate for A * total output
    total_investment_A = max(0.0, s_A * total_output)

    # Initialize investment allocations for each sector/capital type to zero
    Inv_K_T, Inv_K_H, Inv_K_I = 0.0, 0.0, 0.0
    Inv_A_T, Inv_A_H, Inv_A_I = 0.0, 0.0, 0.0  # Inv_A_H will remain 0 by model design

    # --- 2. Allocate K Investment based on MPK ---
    # Proceed only if there are funds to invest in K
    if total_investment_K > epsilon:
        # Consider only non-negative marginal products as drivers for investment
        mpk_T_eff = max(0.0, MPK_T)
        mpk_H_eff = max(0.0, MPK_H)
        mpk_I_eff = max(0.0, MPK_I)

        # Calculate allocation weights: MPK ^ sensitivity
        # Higher sensitivity amplifies differences in returns.
        try:
            weight_K_T = mpk_T_eff**capital_sensitivity
            weight_K_H = mpk_H_eff**capital_sensitivity
            weight_K_I = mpk_I_eff**capital_sensitivity
        except OverflowError:  # Handle extremely large MPK * sensitivity
            # Fallback: Assign weights based on which MPK is max
            max_mpk = max(mpk_T_eff, mpk_H_eff, mpk_I_eff)
            weight_K_T = 1.0 if math.isclose(mpk_T_eff, max_mpk) else 0.0
            weight_K_H = 1.0 if math.isclose(mpk_H_eff, max_mpk) else 0.0
            weight_K_I = 1.0 if math.isclose(mpk_I_eff, max_mpk) else 0.0

        # Sum of weights for normalization
        total_weight_K = weight_K_T + weight_K_H + weight_K_I

        # Allocate proportionally to weights if total weight is positive
        if total_weight_K > epsilon:
            share_K_T = weight_K_T / total_weight_K
            share_K_H = weight_K_H / total_weight_K
            share_K_I = weight_K_I / total_weight_K

            # Distribute the total K investment pool
            Inv_K_T = total_investment_K * share_K_T
            Inv_K_H = total_investment_K * share_K_H
            Inv_K_I = total_investment_K * share_K_I
        # Else: If all MPKs are zero or negative, total_weight_K is zero, and no K investment occurs.

    # --- 3. Allocate A Investment based on MPA ---
    # Proceed only if there are funds to invest in A
    # Note: Sector H does not use A, so MPA_H should be 0 and Inv_A_H will remain 0.
    if total_investment_A > epsilon:
        # Consider only non-negative marginal products
        mpa_T_eff = max(0.0, MPA_T)
        mpa_I_eff = max(0.0, MPA_I)
        # mpa_H_eff = 0.0 (implicitly, as MPA_H should be 0)

        # Calculate allocation weights: MPA ^ sensitivity
        try:
            weight_A_T = mpa_T_eff**capital_sensitivity
            weight_A_I = mpa_I_eff**capital_sensitivity
        except OverflowError:
            max_mpa = max(mpa_T_eff, mpa_I_eff)  # Only compare T and I
            weight_A_T = 1.0 if math.isclose(mpa_T_eff, max_mpa) else 0.0
            weight_A_I = 1.0 if math.isclose(mpa_I_eff, max_mpa) else 0.0

        # Sum of weights for normalization (only for sectors using A)
        total_weight_A = weight_A_T + weight_A_I

        # Allocate proportionally to weights if total weight is positive
        if total_weight_A > epsilon:
            share_A_T = weight_A_T / total_weight_A
            share_A_I = weight_A_I / total_weight_A

            # Distribute the total A investment pool to T and I
            Inv_A_T = total_investment_A * share_A_T
            Inv_A_I = total_investment_A * share_A_I
            # Inv_A_H remains 0.0
        # Else: If MPA_T and MPA_I are zero or negative, no A investment occurs.

    # --- Final Checks and Cleanup ---
    # Ensure non-negativity due to potential tiny float errors
    Inv_K_T = max(0.0, Inv_K_T)
    Inv_K_H = max(0.0, Inv_K_H)
    Inv_K_I = max(0.0, Inv_K_I)
    Inv_A_T = max(0.0, Inv_A_T)
    Inv_A_H = 0.0
    Inv_A_I = max(0.0, Inv_A_I)

    # --- Conservation Check (Optional - helps catch errors) ---
    # Verify that the sum of allocated investments equals the total investment pool.
    allocated_K = Inv_K_T + Inv_K_H + Inv_K_I
    allocated_A = Inv_A_T + Inv_A_H + Inv_A_I  # Inv_A_H is 0

    # If there's a small discrepancy due to floating point math, rescale proportionally.
    if not math.isclose(allocated_K, total_investment_K, rel_tol=1e-7):
        if allocated_K > epsilon:  # Avoid division by zero
            scale_k = total_investment_K / allocated_K
            Inv_K_T *= scale_k
            Inv_K_H *= scale_k
            Inv_K_I *= scale_k
        # Use print statement for debugging if needed:
        # print(f"Warning: K Investment conservation issue. Total: {total_investment_K:.4f}, Allocated: {allocated_K:.4f}. Rescaled.")

    if not math.isclose(allocated_A, total_investment_A, rel_tol=1e-7):
        if allocated_A > epsilon:
            scale_a = total_investment_A / allocated_A
            Inv_A_T *= scale_a
            Inv_A_I *= scale_a  # Inv_A_H remains 0
    # print(f"Warning: A Investment conservation issue. Total: {total_investment_A:.4f}, Allocated: {allocated_A:.4f}. Rescaled.")

    # Return the calculated gross investment amounts for each sector and capital type
    return Inv_K_T, Inv_K_H, Inv_K_I, Inv_A_T, Inv_A_H, Inv_A_I
