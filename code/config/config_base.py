# Calibration for Norway 2025 (Three-Sector Model)

# --- Simulation Settings ---
T_sim = 20  # simulate 20 years (2025–2045)
L_total = 3000000.0  # total labor force ≈ 3.0 million in 2025

# --- AI Adoption (S-curve) Parameters ---
# Logistic adoption curves for AI share (phi) in Traditional and Intelligence sectors.
# phi(t) = phi_init + (phi_max - phi_init) / (1 + exp[-k*(t - t0)])
phi_T_max = 0.7  # max AI share in T (70% long-run automation potential)
phi_T_k = 0.50  # S-curve steepness for T (moderate adoption speed)
phi_T_t0 = 12  # inflection point year for T
phi_T_init = 0.05  # initial AI share in T (5% in 2025)

phi_I_max = 0.95  # max AI share in I (95% potential)
phi_I_k = 0.60  # S-curve steepness for I (faster adoption)
phi_I_t0 = 6  # inflection point year for I
phi_I_init = 0.10  # initial AI share in I (10% in 2025)

# (Human-centric sector H has no direct AI input, φ_H = 0 by assumption.)

# --- Initial Conditions (2025) ---
initial_conditions = {
    # Labor allocation (L_T, L_H, L_I) and Unemployment (L_U)
    "L_T_0": 750000.0,  # T sector labor ≈ 0.75 million (25%) (Industry, Agriculture, Construction)
    "L_H_0": 1530000.0,  # H sector labor ≈ 1.53 million (51%) (Health and Education)
    "L_I_0": 600000.0,  # I sector labor ≈ 0.6 million (20%) (Government, Finance, Research)
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
    "s_K": 0.15,  # 15% of output invested in traditional capital annually
    "s_A": 0.15,  # 15% of output invested in AI capital annually
    # Capital allocation sensitivity:
    "capital_sensitivity": 2.0,  # responsiveness of investment to sectoral returns
}

# --- Production Function Parameters (Nested CES) ---
# Each sector has a nested CES production technology Y = F(K, H) with H = G(A, L).
# 'alpha' = capital share in outer nest; 'rho_outer' and 'rho_inner' are substitution parameters:
#    rho = (σ−1)/σ. rho < 0 ⇒ σ<1 (K and H are complements); rho = 0 ⇒ σ=1 (Cobb-Douglas);
#    rho > 0 ⇒ σ>1 (substitutes)
base_production_params = {
    "T": {  # Traditional sector (manufacturing, oil, etc.)
        "alpha": 0.35,  # capital share in outer nest (K vs H) – higher for capital-intensive sector
        "rho_outer": -0.25,  # substitution between K and H; negative ⇒ complements
        "rho_inner": 0.33,  # substitution between AI and labor; >0 ⇒ some substitutability
        # (Low σ_inner in T reflects limited automation in 2025; AI augments labor more than replacing it.)
    },
    "H": {  # Human-centric sector (health, education, etc.)
        "alpha": 0.25,  # labor-intensive sector (minimal automation)
        "phi_A_share": 0.0,  # AI share fixed 0 (no AI input in H sector)
        "rho_outer": -0.40,  # substitution between K and L in H; mild complements
        "rho_inner": 0.0,  # (unused since φ_A_share=0)
    },
    "I": {  # Intelligence sector (tech, ICT, R&D services)
        "alpha": 0.30,  # capital share ~40% (moderate, since both labor and AI are important)
        "rho_outer": -0.10,  # K vs H substitution in I; complements
        "rho_inner": 0.60,  # AI vs labor substitution;
        # (High σ_inner in I means AI can largely substitute for labor as technology advances)
    },
}

# --- Labor Mobility Parameters ---
# Frictions governing movement of labor between sectors and unemployment.
labor_mobility_params = {
    "mobility_factor": 0.20,  # fraction of desired sector switches that occur per year (20% mobility rate)
    "non_movable_fraction": 0.7,  # fraction of workers who cannot switch sectors (structural immobility)
    "job_finding_rate": 0.70,  # fraction of unemployed finding jobs each year
    "job_separation_rate": 0.1,  # fraction of employed becoming unemployed per year
    "wage_sensitivity": 2.0,  # responsiveness of labor flows to wage differences
}
