#!/usr/bin/env python3
"""Generate main comparison results across key scenarios.

Runs the model for three predefined configurations and produces
publication-quality figures of sectoral outputs, labor allocations,
and wage differentials. Output is saved in ``results/main``.
"""

import os
import importlib

try:
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    HAVE_MPL = False
    plt = None
try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - required dependency
    raise SystemExit(
        "This script requires NumPy. Please install it with 'pip install numpy'."
    ) from exc

from simulation_engine import run_single_simulation

CONFIGS = ["config_noAI", "config_base"]
NAMES_CONFIGS = ["Beskjeden AI adaptasjon", "Rask AI adaptasjon"]
CONFIG_MODULE_PREFIX = "config."
OUTPUT_DIR = os.path.join("../results", "main")


def load_config(name):
    """Dynamically import a configuration module by name."""
    return importlib.import_module(CONFIG_MODULE_PREFIX + name)


def s_curve(phi_init, phi_max, k, t0, years):
    """Return logistic S-curve values for the given parameters."""
    phi = phi_init + (phi_max - phi_init) / (1 + np.exp(-k * (years - t0)))
    phi[0] = phi_init
    return phi


def run_scenario(cfg_name):
    """Run a single scenario and return the results dictionary."""
    cfg = load_config(cfg_name)
    years = np.arange(cfg.T_sim + 1)
    phi_T = s_curve(cfg.phi_T_init, cfg.phi_T_max, cfg.phi_T_k, cfg.phi_T_t0, years)
    phi_I = s_curve(cfg.phi_I_init, cfg.phi_I_max, cfg.phi_I_k, cfg.phi_I_t0, years)

    results = run_single_simulation(
        T_sim=cfg.T_sim,
        L_total=cfg.L_total,
        initial_conditions=cfg.initial_conditions,
        economic_params=cfg.economic_params,
        production_params=cfg.base_production_params,
        labor_mobility_params=cfg.labor_mobility_params,
        phi_T_t=phi_T,
        phi_I_t=phi_I,
        verbose=False,
    )
    return results


def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def save_results(all_results):
    """Save each scenario's results to compressed NumPy files."""
    for name, res in all_results.items():
        path = os.path.join(OUTPUT_DIR, f"{name}_results.npz")
        np.savez(path, **res)


def plot_outputs(all_results):
    if not HAVE_MPL:
        print("Matplotlib not available, skipping output plots.")
        return

    # Calculate global min/max for consistent y-axis ranges
    # Primary axis (sectoral outputs)
    all_sectoral_outputs = []
    for res in all_results.values():
        all_sectoral_outputs.extend([res["Y_T"], res["Y_H"], res["Y_I"]])

    sectoral_min = min(np.min(outputs) for outputs in all_sectoral_outputs)
    sectoral_max = max(np.max(outputs) for outputs in all_sectoral_outputs)

    # Add some padding
    sectoral_range = sectoral_max - sectoral_min
    sectoral_y_min = sectoral_min - 0.05 * sectoral_range
    sectoral_y_max = sectoral_max + 0.05 * sectoral_range

    # Secondary axis (total output)
    all_total_outputs = [res["Y_Total"] for res in all_results.values()]
    total_min = min(np.min(outputs) for outputs in all_total_outputs)
    total_max = max(np.max(outputs) for outputs in all_total_outputs)

    # Add some padding
    total_range = total_max - total_min
    total_y_min = total_min - 0.05 * total_range
    total_y_max = total_max + 0.05 * total_range

    # Create subplots based on number of scenarios
    num_scenarios = len(all_results)
    fig, axes = plt.subplots(
        1, num_scenarios, figsize=(5 * num_scenarios, 4), sharey=False
    )

    # Handle case where there's only one subplot
    if num_scenarios == 1:
        axes = [axes]

    for i, (ax, (name, res)) in enumerate(zip(axes, all_results.items())):
        # Only add labels to the first subplot for the legend
        if i == 0:
            ax.plot(res["years"], res["Y_T"], label="Tradisjonell", linewidth=3)
            ax.plot(res["years"], res["Y_H"], label="Menneskelig", linewidth=3)
            ax.plot(res["years"], res["Y_I"], label="Intelligens", linewidth=3)
        else:
            ax.plot(res["years"], res["Y_T"], linewidth=3)
            ax.plot(res["years"], res["Y_H"], linewidth=3)
            ax.plot(res["years"], res["Y_I"], linewidth=3)

        # Set consistent y-axis range for primary axis
        ax.set_ylim(sectoral_y_min, sectoral_y_max)

        # Create secondary axis for total output
        ax2 = ax.twinx()
        if i == 0:
            ax2.plot(
                res["years"],
                res["Y_Total"],
                label="Total",
                linestyle="--",
                color="black",
            )
        else:
            ax2.plot(res["years"], res["Y_Total"], linestyle="--", color="black")

            # Set consistent y-axis range for secondary axis
        ax2.set_ylim(total_y_min, total_y_max)

        if ax is axes[-1]:
            ax2.set_ylabel("Total produksjon")
        else:
            # Hide y-axis tick labels on secondary axis for non-rightmost subplots
            ax2.set_yticklabels([])

        # Use Norwegian names for titles
        config_index = CONFIGS.index(name)
        ax.set_title(NAMES_CONFIGS[config_index])
        ax.set_xlabel("År")
        if ax is axes[0]:
            ax.set_ylabel("Sektorproduksjon")
        else:
            # Hide y-axis tick labels on primary axis for non-leftmost subplots
            ax.set_yticklabels([])
        ax.grid(True)

    # Primary axis legend
    handles1, labels1 = axes[0].get_legend_handles_labels()
    # Secondary axis legend
    handles2, labels2 = axes[0].get_figure().axes[1].get_legend_handles_labels()
    # Combine legends
    fig.legend(handles1 + handles2, labels1 + labels2, loc="upper center", ncol=4)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(OUTPUT_DIR, "sector_outputs.png"), dpi=300)
    plt.close(fig)


def plot_labor(all_results):
    if not HAVE_MPL:
        print("Matplotlib not available, skipping labor allocation plots.")
        return

    # Calculate global min/max for consistent y-axis ranges
    # Primary axis (sectoral employment)
    all_sectoral_employment = []
    for res in all_results.values():
        all_sectoral_employment.extend([res["L_T"], res["L_H"], res["L_I"]])

    sectoral_min = min(np.min(employment) for employment in all_sectoral_employment)
    sectoral_max = max(np.max(employment) for employment in all_sectoral_employment)

    # Add some padding
    sectoral_range = sectoral_max - sectoral_min
    sectoral_y_min = sectoral_min - 0.05 * sectoral_range
    sectoral_y_max = sectoral_max + 0.05 * sectoral_range

    # Secondary axis (unemployment) - fixed range
    unemployment_y_min = 0
    unemployment_y_max = 250000

    # Create subplots based on number of scenarios
    num_scenarios = len(all_results)
    fig, axes = plt.subplots(
        1, num_scenarios, figsize=(5 * num_scenarios, 4), sharey=False
    )

    # Handle case where there's only one subplot
    if num_scenarios == 1:
        axes = [axes]

    for i, (ax, (name, res)) in enumerate(zip(axes, all_results.items())):
        # Only add labels to the first subplot for the legend
        if i == 0:
            ax.plot(res["years"], res["L_T"], label="L_T", linewidth=3)
            ax.plot(res["years"], res["L_H"], label="L_H", linewidth=3)
            ax.plot(res["years"], res["L_I"], label="L_I", linewidth=3)
        else:
            ax.plot(res["years"], res["L_T"], linewidth=3)
            ax.plot(res["years"], res["L_H"], linewidth=3)
            ax.plot(res["years"], res["L_I"], linewidth=3)

        # Set consistent y-axis range for primary axis
        ax.set_ylim(sectoral_y_min, sectoral_y_max)

        # Create secondary axis for unemployment
        ax2 = ax.twinx()
        if i == 0:
            ax2.plot(
                res["years"],
                res["L_U"],
                label="Arbeidsledige",
                linestyle=":",
                color="darkred",
                alpha=0.5,
            )
        else:
            ax2.plot(
                res["years"], res["L_U"], linestyle=":", color="darkred", alpha=0.5
            )

        # Set consistent y-axis range for secondary axis
        ax2.set_ylim(unemployment_y_min, unemployment_y_max)

        if ax is axes[-1]:
            ax2.set_ylabel("Arbeidsledige")
        else:
            # Hide y-axis tick labels on secondary axis for non-rightmost subplots
            ax2.set_yticklabels([])

        # Use Norwegian names for titles
        config_index = CONFIGS.index(name)
        ax.set_title(NAMES_CONFIGS[config_index])
        ax.set_xlabel("År")
        if ax is axes[0]:
            ax.set_ylabel("Sysselsatte")
        else:
            # Hide y-axis tick labels on primary axis for non-leftmost subplots
            ax.set_yticklabels([])
        ax.grid(True)

    # Primary axis legend
    handles1, labels1 = axes[0].get_legend_handles_labels()
    # Secondary axis legend
    handles2, labels2 = axes[0].get_figure().axes[1].get_legend_handles_labels()
    # Combine legends
    fig.legend(handles1 + handles2, labels1 + labels2, loc="upper center", ncol=4)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(OUTPUT_DIR, "labor_allocation.png"), dpi=300)
    plt.close(fig)


def plot_wage_levels(all_results):
    if not HAVE_MPL:
        print("Matplotlib not available, skipping wage level plots.")
        return

    # Calculate global min/max for consistent y-axis ranges
    all_wages = []
    for res in all_results.values():
        all_wages.extend([res["MPL_T"], res["MPL_H"], res["MPL_I"]])

    global_min = min(np.min(wages) for wages in all_wages)
    global_max = max(np.max(wages) for wages in all_wages)

    # Add some padding
    y_range = global_max - global_min
    y_min = global_min - 0.05 * y_range
    y_max = global_max + 0.05 * y_range

    # Create subplots based on number of scenarios
    num_scenarios = len(all_results)
    fig, axes = plt.subplots(
        1, num_scenarios, figsize=(5 * num_scenarios, 4), sharey=False
    )

    # Handle case where there's only one subplot
    if num_scenarios == 1:
        axes = [axes]

    for i, (ax, (name, res)) in enumerate(zip(axes, all_results.items())):
        # Only add labels to the first subplot for the legend
        if i == 0:
            ax.plot(res["years"], res["MPL_T"], label="w_T", linewidth=3)
            ax.plot(res["years"], res["MPL_H"], label="w_H", linewidth=3)
            ax.plot(res["years"], res["MPL_I"], label="w_I", linewidth=3)
        else:
            ax.plot(res["years"], res["MPL_T"], linewidth=3)
            ax.plot(res["years"], res["MPL_H"], linewidth=3)
            ax.plot(res["years"], res["MPL_I"], linewidth=3)

        # Set consistent y-axis range
        ax.set_ylim(y_min, y_max)

        # Use Norwegian names for titles
        config_index = CONFIGS.index(name)
        ax.set_title(NAMES_CONFIGS[config_index])
        ax.set_xlabel("År")
        if ax is axes[0]:
            ax.set_ylabel("Lønnsnivå")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(OUTPUT_DIR, "wage_levels.png"), dpi=300)
    plt.close(fig)


def plot_wage_diffs(all_results):
    if not HAVE_MPL:
        print("Matplotlib not available, skipping wage differential plots.")
        return

    # Calculate global min/max for consistent y-axis ranges
    all_ratios = []
    for res in all_results.values():
        w_T = res["MPL_T"]
        all_ratios.extend([np.ones_like(w_T), res["MPL_H"] / w_T, res["MPL_I"] / w_T])

    global_min = min(np.min(ratios) for ratios in all_ratios)
    global_max = max(np.max(ratios) for ratios in all_ratios)

    # Add some padding
    y_range = global_max - global_min
    y_min = global_min - 0.05 * y_range
    y_max = global_max + 0.05 * y_range

    # Create subplots based on number of scenarios
    num_scenarios = len(all_results)
    fig, axes = plt.subplots(
        1, num_scenarios, figsize=(5 * num_scenarios, 4), sharey=False
    )

    # Handle case where there's only one subplot
    if num_scenarios == 1:
        axes = [axes]

    for i, (ax, (name, res)) in enumerate(zip(axes, all_results.items())):
        w_T = res["MPL_T"]
        # Only add labels to the first subplot for the legend
        if i == 0:
            ax.plot(
                res["years"],
                np.ones_like(w_T),
                label="w_T / w_T",
                linestyle="--",
                color="black",
                linewidth=3,
            )
            ax.plot(res["years"], res["MPL_H"] / w_T, label="w_H / w_T")
            ax.plot(res["years"], res["MPL_I"] / w_T, label="w_I / w_T")
        else:
            ax.plot(
                res["years"],
                np.ones_like(w_T),
                linestyle="--",
                color="black",
                linewidth=3,
            )
            ax.plot(res["years"], res["MPL_H"] / w_T)
            ax.plot(res["years"], res["MPL_I"] / w_T)

        # Set consistent y-axis range
        ax.set_ylim(y_min, y_max)

        # Use Norwegian names for titles
        config_index = CONFIGS.index(name)
        ax.set_title(NAMES_CONFIGS[config_index])
        ax.set_xlabel("År")
        if ax is axes[0]:
            ax.set_ylabel("Relativ lønn")
        ax.grid(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(OUTPUT_DIR, "wage_differentials.png"), dpi=300)
    plt.close(fig)


def plot_s_curves():
    """Plot the S-curves for phi_T and phi_I across scenarios (excluding config_high)."""
    if not HAVE_MPL:
        print("Matplotlib not available, skipping S-curve plots.")
        return

    # Exclude config_high scenario
    configs_to_plot = [
        cfg for cfg in CONFIGS if cfg != "config_high"  # and cfg != "config_noAI"
    ]

    fig, axes = plt.subplots(1, len(configs_to_plot), figsize=(8, 3), sharey=True)

    # Handle case where there's only one subplot
    if len(configs_to_plot) == 1:
        axes = [axes]

    for col, cfg_name in enumerate(configs_to_plot):
        cfg = load_config(cfg_name)
        years = np.arange(cfg.T_sim + 1)

        # Generate S-curves
        phi_T = s_curve(cfg.phi_T_init, cfg.phi_T_max, cfg.phi_T_k, cfg.phi_T_t0, years)
        phi_I = s_curve(cfg.phi_I_init, cfg.phi_I_max, cfg.phi_I_k, cfg.phi_I_t0, years)

        # Plot both sectors in the same subplot
        if col == 0:
            axes[col].plot(
                years,
                phi_T,
                "-",
                color="forestgreen",
                linewidth=3,
                label="φ_T",
            )
            axes[col].plot(years, phi_I, "--", color="navy", linewidth=3, label="φ_I")
        else:
            axes[col].plot(
                years,
                phi_T,
                "-",
                color="forestgreen",
                linewidth=3,
            )
            axes[col].plot(years, phi_I, "--", color="navy", linewidth=3)

        # Use Norwegian names for titles
        config_index = CONFIGS.index(cfg_name)
        axes[col].set_title(NAMES_CONFIGS[config_index])
        axes[col].set_xlabel("År")
        axes[col].grid(True)
        axes[col].set_ylim(-0.1, 1.1)
        if col == 0:
            axes[col].legend(loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "s_curves.png"), dpi=300)
    plt.close(fig)


def main():
    if HAVE_MPL:
        # plt.style.use("fivethirtyeight")
        plt.style.use("fast")
        # plt.style.use("seaborn-whitegrid")
        # plt.style.use("ggplot")

    else:
        print("Matplotlib not installed; skipping all plots.")

    ensure_output_dir()
    all_results = {name: run_scenario(name) for name in CONFIGS}

    save_results(all_results)
    plot_outputs(all_results)
    plot_labor(all_results)
    plot_wage_levels(all_results)
    plot_wage_diffs(all_results)
    plot_s_curves()


if __name__ == "__main__":
    main()
