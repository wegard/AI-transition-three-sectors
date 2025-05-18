# Calibration for Norway 2025 (Three-Sector Model)

# --- Simulation Settings ---
T_sim = 12  # simulate 12 years (2025–2037)
L_total = 3000000.0  # total labor force ≈ 3.0 million in 2025

# --- AI Adoption (S-curve) Parameters ---
# Logistic adoption curves for AI share (phi) in Traditional and Intelligence sectors.
# phi(t) = phi_init + (phi_max - phi_init) / (1 + exp[-k*(t - t0)])
phi_T_max = 0.99  # max AI share in T (90% long-run automation potential)
phi_T_k = 0.80  # S-curve steepness for T (moderate adoption speed)
phi_T_t0 = 5  # inflection point year for T
phi_T_init = 0.1  # initial AI share in T (15% in 2025)

phi_I_max = 0.99  # max AI share in I (95% potential)
phi_I_k = 0.80  # S-curve steepness for I (faster adoption)
phi_I_t0 = 5  # inflection point year for I
phi_I_init = 0.1  # initial AI share in I (15% in 2025)

# (Human-centric sector H has no direct AI input, φ_H = 0 by assumption.)

# --- Initial Conditions (2025) ---
initial_conditions = {
    # Labor allocation (L_T, L_H, L_I) and Unemployment (L_U)
    "L_T_0": 600000.0,  # T sector labor ≈ 0.6 million (20%) (Industry, Agriculture, Construction)
    "L_H_0": 1140000.0,  # H sector labor ≈ 1.2 million (38%) (Health and Education)
    "L_I_0": 1140000.0,  # I sector labor ≈ 1.2 million (38%) (Government, Finance, Research)
    "L_U_0": 120000.0,  # Unemployed ≈ 0.12 million (4% unemployment)
    # Initial capital stocks by sector (K = traditional capital, A = AI capital)
    # Total K ≈ 20 trillion NOK split across sectors (T is most capital-intensive).
    "K_T_0": 9700.0,  # Traditional capital ≈ 9.7 trillion NOK (heavy industry, oil, etc.)
    "K_H_0": 6500.0,  # Human sector capital ≈ 6.5 trillion NOK (infrastructure for health, edu)
    "K_I_0": 3900.0,  # Intelligence capital ≈ 3.9 trillion NOK (tech, R&D, etc.)
    # Initial AI-specific capital A (subset of K):
    # Limited AI deployment in 2025: ~10% of K_T, 0% of K_H, ~25% of K_I
    "A_T_0": 1455.0,  # AI capital in T ≈ 1.455 trillion NOK (15% of K_T)
    "A_H_0": 0.0,  # AI capital in H = 0 (H sector has no AI in production)
    "A_I_0": 975.0,  # AI capital in I ≈ 0.975 trillion NOK (25% of K_I)
}

# --- Economic Parameters ---
economic_params = {
    # Depreciation rates (annual):
    "delta_K": 0.05,  # 5% depreciation of traditional capital per year
    "delta_A": 0.05,  # 5% depreciation of AI capital per year
    # Aggregate investment/savings rates (fraction of total output):
    "s_K": 0.15,  # ~15% of output invested in traditional capital annually
    "s_A": 0.25,  # ~25% of output invested in AI capital annually
    # Capital allocation sensitivity:
    "capital_sensitivity": 3.5,  # responsiveness of investment to sectoral returns
}

# --- Production Function Parameters (Nested CES) ---
# Each sector has a nested CES production technology Y = F(K, H) with H = G(A, L).
# 'alpha' = capital share in outer nest; 'rho_outer' and 'rho_inner' are substitution parameters:
#    rho = (σ−1)/σ. rho < 0 ⇒ σ<1 (K and H are complements); rho = 0 ⇒ σ=1 (Cobb-Douglas);
#    rho > 0 ⇒ σ>1 (substitutes)
base_production_params = {
    "T": {  # Traditional sector (manufacturing, oil, etc.)
        "alpha": 0.50,  # capital share in outer nest (K vs H) – higher (60%) for capital-intensive sector
        "rho_outer": -0.20,  # substitution between K and H; negative ⇒ K and (A,L) are complements (σ_outer≈0.83)
        "rho_inner": -0.10,  # substitution between AI and labor; slightly negative ⇒ A and L are complements (σ_inner≈0.91)
        # (Low σ_inner in T reflects limited automation in 2025 – AI augments labor rather than replacing it.)
    },
    "H": {  # Human-centric sector (health, education, etc.)
        "alpha": 0.50,  # capital share ~50% (labor share ~50%) – labor-intensive sector (minimal automation)
        "phi_A_share": 0.0,  # AI share fixed 0 (no AI input in H sector)
        "rho_outer": -0.15,  # substitution between K and L in H; σ≈0.87 (mild complements)
        "rho_inner": 0.0,  # (unused since φ_A_share=0)
    },
    "I": {  # Intelligence sector (tech, ICT, R&D services)
        "alpha": 0.50,  # capital share ~40% (moderate, since both labor and AI are important)
        "rho_outer": -0.20,  # K vs H substitution in I; complements (σ_outer≈0.83)
        "rho_inner": 0.999,  # AI vs labor substitution; ~0.999 → σ_inner ≈ 1000 (nearly perfect substitutes)
        # (High σ_inner in I means AI can largely substitute for labor as technology advances)
    },
}

# --- Labor Mobility Parameters ---
# Frictions governing movement of labor between sectors and unemployment.
labor_mobility_params = {
    "mobility_factor": 0.3,  # fraction of desired sector switches that occur per year (30% mobility rate)
    "non_movable_fraction": 0.6,  # fraction of workers who cannot switch sectors (structural immobility)
    "job_finding_rate": 0.4,  # fraction of unemployed finding jobs each year
    "job_separation_rate": 0.30,  # fraction of employed becoming unemployed per year
    "wage_sensitivity": 5.0,  # responsiveness of labor flows to wage differences
}
