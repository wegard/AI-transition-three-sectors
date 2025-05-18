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

CONFIGS = ["config_noAI", "config_base", "config_high"]
CONFIG_MODULE_PREFIX = "config."
OUTPUT_DIR = os.path.join("results", "main")


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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (name, res) in zip(axes, all_results.items()):
        ax.plot(res["years"], res["Y_T"], label="Traditional")
        ax.plot(res["years"], res["Y_H"], label="Human")
        ax.plot(res["years"], res["Y_I"], label="Intelligence")
        ax.plot(
            res["years"], res["Y_Total"], label="Total", linestyle="--", color="black"
        )
        ax.set_title(name)
        ax.set_xlabel("Year")
        if ax is axes[0]:
            ax.set_ylabel("Output")
        ax.grid(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(OUTPUT_DIR, "sector_outputs.png"), dpi=300)
    plt.close(fig)


def plot_labor(all_results):
    if not HAVE_MPL:
        print("Matplotlib not available, skipping labor allocation plots.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (name, res) in zip(axes, all_results.items()):
        ax.plot(res["years"], res["L_T"], label="L_T")
        ax.plot(res["years"], res["L_H"], label="L_H")
        ax.plot(res["years"], res["L_I"], label="L_I")
        ax.plot(res["years"], res["L_U"], label="L_U")
        ax.set_title(name)
        ax.set_xlabel("Year")
        if ax is axes[0]:
            ax.set_ylabel("Labor units")
        ax.grid(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(OUTPUT_DIR, "labor_allocation.png"), dpi=300)
    plt.close(fig)


def plot_wage_diffs(all_results):
    if not HAVE_MPL:
        print("Matplotlib not available, skipping wage differential plots.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (name, res) in zip(axes, all_results.items()):
        w_T = res["MPL_T"]
        ax.plot(
            res["years"],
            np.ones_like(w_T),
            label="w_T / w_T",
            linestyle="--",
            color="black",
        )
        ax.plot(res["years"], res["MPL_H"] / w_T, label="w_H / w_T")
        ax.plot(res["years"], res["MPL_I"] / w_T, label="w_I / w_T")
        ax.set_title(name)
        ax.set_xlabel("Year")
        if ax is axes[0]:
            ax.set_ylabel("Relative wage")
        ax.grid(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(OUTPUT_DIR, "wage_differentials.png"), dpi=300)
    plt.close(fig)


def main():
    if HAVE_MPL:
        plt.style.use("seaborn-v0_8-whitegrid")
    else:
        print("Matplotlib not installed; skipping all plots.")

    ensure_output_dir()
    all_results = {name: run_scenario(name) for name in CONFIGS}

    save_results(all_results)
    plot_outputs(all_results)
    plot_labor(all_results)
    plot_wage_diffs(all_results)


if __name__ == "__main__":
    main()
