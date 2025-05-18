# sympy_model_derivation.py

import sympy

# --- Define Symbolic Variables ---
# Define inputs (Capital K, AI Capital A, Labor L) as positive real symbols
K, A, L = sympy.symbols("K A L", positive=True, real=True)

# Define parameters as real symbols
# alpha: Share parameter for K in the outer CES nest (0 < alpha < 1)
# phi_A: Share parameter for A within the inner A-L nest (0 < phi_A < 1), represents phi_A_share
# rho_o: Substitution parameter for the outer K-H nest (rho_o <= 1), represents rho_outer
# rho_i: Substitution parameter for the inner A-L nest (rho_i <= 1), represents rho_inner
alpha, phi_A, rho_o, rho_i = sympy.symbols("alpha phi_A rho_o rho_i", real=True)

# Print symbol definitions and assumptions
print("--- Symbol Definitions and Assumptions ---")
print(f"Inputs (positive, real): K={K}, A={A}, L={L}")
print(f"Parameters (real): alpha={alpha}, phi_A={phi_A}, rho_o={rho_o}, rho_i={rho_i}")
print(
    "  Assumed practical ranges: 0 < alpha < 1, 0 <= phi_A <= 1, rho_o <= 1, rho_i <= 1"
)
print("-" * 50)

# --- Define the Nested CES Production Function Symbolically ---

# 1. Inner Aggregate H = g(A, L)
# H = [ phi_A * A^rho_i + (1 - phi_A) * L^rho_i ]^(1 / rho_i)
print("Defining Inner Aggregate H...")
# Use sympy.Pow for exponents, including symbolic ones
inner_base = phi_A * sympy.Pow(A, rho_i) + (1 - phi_A) * sympy.Pow(L, rho_i)
# Use standard division for the symbolic exponent 1/rho_i
H_expr = sympy.Pow(inner_base, 1 / rho_i)

print("H(A, L) = H_expr")
# sympy.pprint(H_expr)
print("-" * 50)

# 2. Outer Production Function Y = f(K, H)
# Y = [ alpha * K^rho_o + (1 - alpha) * H^rho_o ]^(1 / rho_o)
print("Defining Outer Production Function Y...")
outer_base = alpha * sympy.Pow(K, rho_o) + (1 - alpha) * sympy.Pow(H_expr, rho_o)
# Use standard division for the symbolic exponent 1/rho_o
Y_expr = sympy.Pow(outer_base, 1 / rho_o)

print("Y(K, A, L) = Y_expr")
# sympy.pprint(Y_expr)
print("-" * 50)

# --- Calculate Marginal Products Symbolically using Differentiation ---

print("Calculating Marginal Products...")

# 1. Marginal Product of Traditional Capital (MPK) = dY/dK
print("\nCalculating MPK = dY/dK...")
# Differentiate the full Y expression with respect to K
MPK_expr_raw = sympy.diff(Y_expr, K)
# Simplify the resulting expression
MPK_expr = sympy.simplify(MPK_expr_raw)
# Optionally use powsimp for further simplification of powers
MPK_expr = sympy.powsimp(MPK_expr, force=True)

print("MPK = MPK_expr")
# sympy.pprint(MPK_expr)
print("-" * 50)

# 2. Marginal Product of AI Capital (MPA) = dY/dA
# Use the chain rule: MPA = (dY/dH) * (dH/dA)

# 2a. Calculate dH/dA
print("\nCalculating dH/dA...")
dH_dA_expr_raw = sympy.diff(H_expr, A)
dH_dA_expr = sympy.simplify(dH_dA_expr_raw)
dH_dA_expr = sympy.powsimp(dH_dA_expr, force=True)

print("dH/dA = dH_dA_expr")
# sympy.pprint(dH_dA_expr)
print("-" * 50)

# 2b. Calculate dY/dH
# It's easier to define Y with an intermediate H symbol for this derivative
print("\nCalculating dY/dH...")
H_sym = sympy.symbols("H", positive=True, real=True)  # Intermediate symbol for H
# Define Y again, but using H_sym instead of the full H_expr
Y_with_H_sym = sympy.Pow(
    alpha * sympy.Pow(K, rho_o) + (1 - alpha) * sympy.Pow(H_sym, rho_o), 1 / rho_o
)
dY_dH_sym_expr_raw = sympy.diff(Y_with_H_sym, H_sym)
dY_dH_sym_expr = sympy.simplify(dY_dH_sym_expr_raw)
dY_dH_sym_expr = sympy.powsimp(dY_dH_sym_expr, force=True)

# Substitute the full expression for H (H_expr) back into the derivative dY/dH
dY_dH_final = dY_dH_sym_expr.subs(H_sym, H_expr)

print("dY/dH (with H substituted back) = ")
# sympy.pprint(dY_dH_final)

# 2c. Calculate MPA = (dY/dH) * (dH/dA)
print("\nCalculating MPA = (dY/dH) * (dH/dA)...")
MPA_expr_raw = dY_dH_final * dH_dA_expr
MPA_expr = sympy.simplify(MPA_expr_raw)
MPA_expr = sympy.powsimp(MPA_expr, force=True)

print("MPA = MPA_expr")
# sympy.pprint(MPA_expr)
print("-" * 50)

# 3. Marginal Product of Labor (MPL) = dY/dL
# Use the chain rule: MPL = (dY/dH) * (dH/dL)

# 3a. Calculate dH/dL
print("\nCalculating dH/dL...")
dH_dL_expr_raw = sympy.diff(H_expr, L)
dH_dL_expr = sympy.simplify(dH_dL_expr_raw)
dH_dL_expr = sympy.powsimp(dH_dL_expr, force=True)

print("dH/dL = dH_dL_expr")
# sympy.pprint(dH_dL_expr)

# 3b. Calculate MPL = (dY/dH) * (dH/dL)
# We already have dY/dH_final from the MPA calculation
print("\nCalculating MPL = (dY/dH) * (dH/dL)...")
MPL_expr_raw = dY_dH_final * dH_dL_expr
MPL_expr = sympy.simplify(MPL_expr_raw)
MPL_expr = sympy.powsimp(MPL_expr, force=True)

print("MPL = MPL_expr")
# sympy.pprint(MPL_expr)
print("-" * 50)

# --- (Optional) Define Elasticities of Substitution ---
print("Elasticities of Substitution (Sigma):")
sigma_o = 1 / (1 - rho_o)
sigma_i = 1 / (1 - rho_i)

print("sigma_outer = sigma_o")
# sympy.pprint(sigma_o)
print("\nsigma_inner = sigma_i")
# sympy.pprint(sigma_i)
print("-" * 50)

print("Symbolic derivation complete.")

# --- Example Usage (Commented Out) ---
# params_example = {alpha: 0.4, phi_A: 0.1, rho_o: -0.4, rho_i: -0.2}
# MPK_val_expr = MPK_expr.subs(params_example)
# print("\nMPK expression with example parameters:")
# sympy.pprint(MPK_val_expr)
# MPK_num = MPK_val_expr.subs({K: 100, A: 10, L: 500}).evalf()
# print(f"\nExample numerical MPK: {MPK_num}")
