# run_experiments.py
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy
import re
import os  # Import the os module for path manipulation
import pickle
from collections.abc import Iterable  # To check for list/array types

####################################################
# --- Import simulation engine and configuration ---
####################################################
try:
    # Import the function that runs a single simulation instance
    from simulation_engine import run_single_simulation

    # Import the configuration file holding baseline parameters
    import config.config_noAI as config

    print("Successfully imported simulation engine and config.")
except ImportError as e:
    # Handle potential import errors gracefully
    print("\n" + "=" * 50)
    print(f"Error: Could not import required module: {e}")
    print(
        "Ensure simulation_engine.py and config.py exist in the same directory and have no errors."
    )
    print("=" * 50 + "\n")
    sys.exit(1)  # Exit the script if imports fail


######################################################
# --- Helper Functions for Nested Parameter Access ---
# These functions allow accessing and modifying values
# within potentially nested dictionaries (like the parameter
# dictionaries loaded from config.py) using a string path.
# This avoids using potentially unsafe 'eval' and provides
# better error handling.
######################################################
def get_nested_value(data_dict, path_str):
    """
    Gets a value from a nested dictionary using a string path.

    Example: get_nested_value(params, "base_production_params['I']['rho_inner']")

    Args:
        data_dict (dict): The dictionary to search within.
        path_str (str): A string representing the path, using standard Python
                      dictionary and list access syntax (e.g., "dict['key1'][key2]").

    Returns:
        The value at the specified path, or None if the path is invalid or an error occurs.
    """
    # Use regex to find all keys specified in the path string (e.g., ['key'] or .key)
    # This handles both dictionary keys (strings) and potentially list indices if needed later.
    keys = re.findall(r"\['(.*?)'\]|\.(.*?)", path_str)
    current_level = data_dict  # Start at the top level of the dictionary
    try:
        # Iterate through the sequence of keys found in the path
        for key_tuple in keys:
            # The regex returns tuples like ('key', '') or ('', 'key'); get the actual key
            key = next(k for k in key_tuple if k)
            # Check if the current level is a dictionary before attempting to access the key
            if isinstance(current_level, dict):
                current_level = current_level[key]  # Move deeper into the dictionary
            else:
                # Raise an error if trying to access a key on a non-dictionary item
                raise TypeError(
                    f"Path element '{key}' requires a dictionary, but found {type(current_level)}"
                )
        # Return the final value found at the end of the path
        return current_level
    except (KeyError, TypeError, IndexError) as e:
        # Handle errors like invalid keys or incorrect types along the path
        print(f"Error accessing path '{path_str}': {e}")
        return None  # Return None to indicate failure


def set_nested_value(data_dict, path_str, value):
    """
    Sets a value in a nested dictionary using a string path. Modifies the dictionary in place.

    Example: set_nested_value(params, "base_production_params['I']['rho_inner']", 0.8)

    Args:
        data_dict (dict): The dictionary to modify.
        path_str (str): The string path specifying where to set the value.
        value: The value to set at the specified path.

    Returns:
        True if the value was successfully set, False otherwise.
    """
    # Find all keys in the path string
    keys = re.findall(r"\['(.*?)'\]|\.(.*?)", path_str)
    current_level = data_dict  # Start at the top level
    try:
        # Iterate through the keys, stopping before the last one
        for i, key_tuple in enumerate(keys):
            key = next(k for k in key_tuple if k)  # Extract the key
            if i == len(keys) - 1:  # If this is the last key in the path
                # Ensure the current level is a dictionary before setting the value
                if isinstance(current_level, dict):
                    current_level[key] = value  # Set the value
                    return True  # Return True indicating success
                else:
                    # Raise error if the parent is not a dictionary
                    raise TypeError(
                        f"Cannot set value at '{key}', parent is not a dictionary."
                    )
            else:  # If not the last key, traverse deeper
                # Ensure the current level is a dictionary before accessing the next key
                if isinstance(current_level, dict):
                    current_level = current_level[key]  # Move to the next level
                else:
                    # Raise error if the path expects a dictionary but finds something else
                    raise TypeError(
                        f"Path element '{key}' requires a dictionary, but found {type(current_level)}"
                    )
    except (KeyError, TypeError, IndexError) as e:
        # Handle errors during path traversal or setting
        print(f"Error setting path '{path_str}': {e}")
        return False  # Return False indicating failure


##################################
# --- Main Experiment Function ---
##################################
def run_parameter_experiment(
    param_path_str, param_values, experiment_name="default_experiment", do_plots=True
):
    """
    Runs a simulation experiment by varying a specified parameter and saves
    output (plots, results dictionary) into a dedicated subfolder within a
    main 'experiments' directory.

    Args:
        param_path_str (str): String representing the path to the parameter to vary within the
                              config structure (e.g., "economic_params['s_K']", "base_production_params['I']['rho_inner']").
        param_values (list or np.array): A list or array of values to iterate the parameter over for the sensitivity analysis.
        experiment_name (str): A descriptive name for this specific experiment. This name is used
                               to create the output subfolder within the main 'experiments' directory.
                               Defaults to 'default_experiment'.

    Returns:
        dict: A dictionary containing the simulation results for the baseline run and all
              successful parameter variation runs. Returns None if input validation fails or
              directory creation fails. Returns only baseline results if no variations succeed.
    """
    # --- Define Main Output Directory ---
    main_output_dir = "../results/experiments"  # Name of the top-level directory to store all experiment outputs.

    # --- Input Validation ---
    # Check if param_values is a valid sequence (list, array, tuple, etc.) and not empty
    if (
        not isinstance(param_values, Iterable)
        or isinstance(param_values, str)  # Ensure it's not just a string
        or len(param_values) == 0
    ):
        print("Error: param_values must be a non-empty list or numpy array.")
        return None  # Exit function if invalid
    # Check if the parameter path string is valid
    if not isinstance(param_path_str, str) or not param_path_str:
        print("Error: param_path_str must be a non-empty string.")
        return None
    # Check if the experiment name is valid
    if not isinstance(experiment_name, str) or not experiment_name:
        print("Error: experiment_name must be a non-empty string.")
        return None

    # Sanitize the experiment name to make it suitable for a directory name
    # Replace characters that are problematic in file paths with underscores
    safe_experiment_name = re.sub(r"[^\w\-_\. ]", "_", experiment_name)
    if safe_experiment_name != experiment_name:
        # Inform the user if the name was changed
        print(
            f"Warning: Experiment name sanitized to '{safe_experiment_name}' for directory creation."
        )
        experiment_name = safe_experiment_name  # Use the sanitized name
    # Update the output directory path with the potentially sanitized name
    experiment_output_dir = os.path.join(main_output_dir, experiment_name)

    # --- Create Output Directories ---
    try:
        # Create the main 'experiments' directory if it doesn't already exist. 'exist_ok=True' prevents an error if it's already there.
        os.makedirs(main_output_dir, exist_ok=True)
        # Create the specific subdirectory for this experiment within 'experiments'.
        os.makedirs(experiment_output_dir, exist_ok=True)
        print(f"Ensured output directory exists: '{experiment_output_dir}'")
    except OSError as e:
        # Handle potential errors during directory creation (e.g., permissions issues)
        print(f"Error creating output directories: {e}")
        return None  # Exit function if directories can't be created

    # --- Print Experiment Setup (after sanitization & directory creation) ---
    print("\n" + "=" * 60)
    print(f"Starting Experiment: '{experiment_name}'")
    print(f"Varying Parameter: '{param_path_str}'")
    print(f"Values to test: {param_values}")
    print(f"Output will be saved in: '{experiment_output_dir}'")
    print("=" * 60)

    # --- Load Fixed Parameters & Generate Dependent Vars from Config ---
    # Use deep copies to ensure that modifications made during parameter variations
    # do not affect the original config module data or subsequent experiments.
    T_sim = copy.deepcopy(config.T_sim)
    L_total = copy.deepcopy(config.L_total)
    years = np.arange(T_sim + 1)  # Create array of years for simulation and plotting

    # Generate the S-shaped AI adoption curves based on parameters from the config file
    phi_T_t = config.phi_T_init + (config.phi_T_max - config.phi_T_init) / (
        1 + np.exp(-config.phi_T_k * (years - config.phi_T_t0))
    )
    phi_T_t[0] = config.phi_T_init  # Ensure starting value is exactly phi_T_init
    phi_I_t = config.phi_I_init + (config.phi_I_max - config.phi_I_init) / (
        1 + np.exp(-config.phi_I_k * (years - config.phi_I_t0))
    )
    phi_I_t[0] = config.phi_I_init  # Ensure starting value is exactly phi_I_init

    # Load parameter dictionaries directly from the imported config module
    initial_conditions = copy.deepcopy(config.initial_conditions)
    economic_params = copy.deepcopy(config.economic_params)
    base_production_params = copy.deepcopy(
        config.base_production_params
    )  # Store the baseline production params
    labor_mobility_params = copy.deepcopy(config.labor_mobility_params)

    # --- Further Validation: Check parameter path validity and type compatibility ---
    phi_T_keys = ["phi_T_max", "phi_T_k", "phi_T_t0", "phi_T_init"]
    phi_I_keys = ["phi_I_max", "phi_I_k", "phi_I_t0", "phi_I_init"]
    phi_param_type = None  # 'T' or 'I' if varying adoption curve parameters
    top_level_key = None

    if param_path_str in phi_T_keys:
        phi_param_type = "T"
        original_value = getattr(config, param_path_str)
    elif param_path_str in phi_I_keys:
        phi_param_type = "I"
        original_value = getattr(config, param_path_str)
    else:
        # Create a temporary combined dictionary of all base parameters for easier validation access
        all_base_params = {
            "initial_conditions": initial_conditions,
            "economic_params": economic_params,
            "base_production_params": base_production_params,
            "labor_mobility_params": labor_mobility_params,
        }
        # Determine the top-level dictionary key from the path string (e.g., 'economic_params')
        top_level_key = param_path_str.split("[")[0].split(".")[0]
        if top_level_key not in all_base_params:
            # Check if the top-level key exists in our combined structure
            print(
                f"Error: Invalid parameter path. Top-level key '{top_level_key}' not found in config dictionaries."
            )
            return None

        # Try to retrieve the original value using the helper function to check path validity
        original_value = get_nested_value(
            all_base_params[top_level_key], param_path_str
        )
        if original_value is None:
            # If get_nested_value returned None, the path was likely invalid
            print(
                f"Error: Could not retrieve original value for parameter path '{param_path_str}'. Path might be invalid."
            )
            return None

    # Check if the type of the values provided for variation matches the original parameter's type
    first_test_value = param_values[0]
    if type(original_value) != type(first_test_value):
        # Allow integers and floats to be considered compatible types
        if not (
            isinstance(original_value, (int, float))
            and isinstance(first_test_value, (int, float))
        ):
            print(
                f"Error: Type mismatch. Parameter '{param_path_str}' has type {type(original_value)}, but test values have type {type(first_test_value)}."
            )
            return None

    print(
        f"Parameter '{param_path_str}' found. Original value: {original_value} (Type: {type(original_value)}). Test values type compatible."
    )
    print("-" * 60)

    # --- Run Baseline Simulation ---
    # Run the simulation once with the original, unmodified parameters from config.py
    # This serves as a reference point for comparison.
    print("Running Baseline Simulation (using original config values)...")
    baseline_start_time = time.time()  # Start timing
    # Use the original parameter dicts loaded directly from config
    baseline_results = run_single_simulation(
        T_sim=T_sim,
        L_total=L_total,
        initial_conditions=config.initial_conditions,  # Use original config
        economic_params=config.economic_params,  # Use original config
        production_params=config.base_production_params,  # Use original config
        labor_mobility_params=config.labor_mobility_params,  # Use original config
        phi_T_t=phi_T_t,
        phi_I_t=phi_I_t,
        verbose=False,  # Suppress detailed output from the simulation engine
    )
    baseline_end_time = time.time()  # End timing
    print(
        f"Baseline simulation finished in {baseline_end_time - baseline_start_time:.2f} seconds."
    )
    print("--- Baseline Final State ---")
    # Print a summary of key final values from the baseline run
    print(f"  Total Output: {baseline_results['Y_Total'][-1]:.2f}")
    print(
        f"  Labor (T, H, I, U): {baseline_results['L_T'][-1]:.1f}, {baseline_results['L_H'][-1]:.1f}, {baseline_results['L_I'][-1]:.1f}, {baseline_results['L_U'][-1]:.1f}"
    )
    print(
        f"  Wages (T, H, I): {baseline_results['MPL_T'][-1]:.3f}, {baseline_results['MPL_H'][-1]:.3f}, {baseline_results['MPL_I'][-1]:.3f}"
    )
    print("-" * 28)

    # --- Experiment Execution ---
    # Now, loop through the provided parameter values and run the simulation for each variation.
    print("\nStarting Parameter Variation Simulations...")
    all_results = {}  # Initialize dictionary to store results for all runs
    all_results["baseline"] = (
        baseline_results  # Add the baseline results for easy plotting comparison
    )
    exp_start_time = time.time()  # Start timing the variations

    # Extract a short name for labels/filenames
    param_key_list = re.findall(r"\['(.*?)'\]", param_path_str)
    if param_key_list:
        param_short_name = param_key_list[-1]
    else:
        param_short_name = param_path_str

    # Loop through each value provided in param_values
    for i, p_value in enumerate(param_values):
        # Create a unique label for this specific simulation run (e.g., "rho_inner=0.5")
        run_label = f"{param_short_name}={p_value:.3g}"  # Using .3g for flexible number formatting
        print(f"\nRunning Simulation {i+1}/{len(param_values)}: {run_label}...")

        # --- Create Deep Copies of Parameter Dictionaries for This Run ---
        # Deep copy to avoid carry-over between runs
        current_initial_conditions = copy.deepcopy(initial_conditions)
        current_economic_params = copy.deepcopy(economic_params)
        current_production_params = copy.deepcopy(base_production_params)
        current_labor_mobility_params = copy.deepcopy(labor_mobility_params)

        phi_T_t_run = phi_T_t
        phi_I_t_run = phi_I_t

        if phi_param_type is None:
            # Modify dictionary-based parameters
            current_params_all = {
                "initial_conditions": current_initial_conditions,
                "economic_params": current_economic_params,
                "base_production_params": current_production_params,
                "labor_mobility_params": current_labor_mobility_params,
            }
            current_dict_to_modify = current_params_all[top_level_key]
            if not set_nested_value(current_dict_to_modify, param_path_str, p_value):
                print(
                    f"  ERROR: Failed to set parameter for value {p_value}. Skipping this run."
                )
                continue
            print(f"  Set '{param_path_str}' = {p_value}")
        else:
            # Recompute the adoption curve with the modified parameter
            if phi_param_type == "T":
                phi_T_max = (
                    p_value if param_path_str == "phi_T_max" else config.phi_T_max
                )
                phi_T_k = p_value if param_path_str == "phi_T_k" else config.phi_T_k
                phi_T_t0 = p_value if param_path_str == "phi_T_t0" else config.phi_T_t0
                phi_T_init = (
                    p_value if param_path_str == "phi_T_init" else config.phi_T_init
                )
                phi_T_t_run = phi_T_init + (phi_T_max - phi_T_init) / (
                    1 + np.exp(-phi_T_k * (years - phi_T_t0))
                )
                phi_T_t_run[0] = phi_T_init
            elif phi_param_type == "I":
                phi_I_max = (
                    p_value if param_path_str == "phi_I_max" else config.phi_I_max
                )
                phi_I_k = p_value if param_path_str == "phi_I_k" else config.phi_I_k
                phi_I_t0 = p_value if param_path_str == "phi_I_t0" else config.phi_I_t0
                phi_I_init = (
                    p_value if param_path_str == "phi_I_init" else config.phi_I_init
                )
                phi_I_t_run = phi_I_init + (phi_I_max - phi_I_init) / (
                    1 + np.exp(-phi_I_k * (years - phi_I_t0))
                )
                phi_I_t_run[0] = phi_I_init
            print(f"  Set '{param_path_str}' = {p_value}")

        # --- Run the Simulation ---
        try:
            # Call the simulation engine with the modified parameter dictionaries for this run
            simulation_output = run_single_simulation(
                T_sim=T_sim,
                L_total=L_total,
                initial_conditions=current_initial_conditions,  # Pass the modified dict
                economic_params=current_economic_params,  # Pass the modified dict
                production_params=current_production_params,  # Pass the modified dict
                labor_mobility_params=current_labor_mobility_params,  # Pass the modified dict
                phi_T_t=phi_T_t_run,
                phi_I_t=phi_I_t_run,
                verbose=False,  # Keep detailed engine output off during loops
            )
            # --- Store Results ---
            # Store the output dictionary (containing time series data) using the run_label as the key
            all_results[run_label] = simulation_output
            print(
                f"Finished run {run_label}. Final Total Output: {simulation_output['Y_Total'][-1]:.2f}"
            )
        except Exception as e:
            # Catch any errors that might occur during the simulation itself
            print(f"  ERROR during simulation run for {run_label}: {e}. Skipping.")
            continue  # Skip this run if the simulation fails

    # --- Timing and Completion Message ---
    exp_end_time = time.time()
    print("-" * 60)
    print(
        f"All variation simulations completed in {exp_end_time - exp_start_time:.2f} seconds."
    )
    print("-" * 60)

    if do_plots:
        ####################################
        # --- Analysis and Visualization ---
        ####################################
        # Check if there are any successful variation results to plot (besides the baseline)
        if not all_results or len(all_results) <= 1:
            print("No successful variation simulations to plot.")
            return (
                all_results  # Return just the baseline results if nothing else worked
            )

        # --- Plotting Setup ---
        print(f"Generating comparison plots in '{experiment_output_dir}'...")
        plt.style.use("seaborn-v0_8-darkgrid")  # Use a visually appealing plot style
        # Create a label for the legend title based on the parameter varied
        plot_param_label = f"{param_short_name} value"
        # Create a base filename for plots, incorporating the experiment name and varied parameter
        # Replace potentially problematic characters for filenames
        plot_filename_base = f"exp_{experiment_name}_vs_{param_short_name.replace('/', '_').replace(' ', '')}"

        # --- Plot Generation ---
        #########################

        # Plot: Sector Labor Allocations (L_I, L_H, L_T) in separate panels
        fig_lab, axes_lab = plt.subplots(
            3, 1, figsize=(8, 10), sharex=True
        )  # 3 rows, 1 col, shared X-axis
        # Iterate through all results (including baseline)
        for run_label, results_data in all_results.items():
            # Default style for variation runs
            style = {"marker": ".", "markersize": 5}
            # Special style for the baseline run to make it distinct
            if run_label == "baseline":
                style = {
                    "lw": 2,
                    "color": "black",
                    "ls": "--",
                    "marker": "x",
                    "markersize": 4,
                }
            # Plot data on respective axes
            axes_lab[0].plot(
                results_data["years"], results_data["L_I"], label=run_label, **style
            )
            axes_lab[1].plot(
                results_data["years"], results_data["L_H"], label=run_label, **style
            )
            axes_lab[2].plot(
                results_data["years"], results_data["L_T"], label=run_label, **style
            )
        # Set titles, labels, and grids for each subplot
        axes_lab[0].set_title(f"Intelligence Sector Labor ($L_I$)")
        axes_lab[0].set_ylabel("Labor Units ($L_I$)")
        axes_lab[0].grid(True)
        axes_lab[1].set_title(f"Human Sector Labor ($L_H$)")
        axes_lab[1].set_ylabel("Labor Units ($L_H$)")
        axes_lab[1].grid(True)
        axes_lab[2].set_title(f"Traditional Sector Labor ($L_T$)")
        axes_lab[2].set_ylabel("Labor Units ($L_T$)")
        axes_lab[2].set_xlabel("Year")
        axes_lab[2].grid(True)
        # Create a single legend for the entire figure, placed outside the plot area
        fig_lab.legend(
            title=plot_param_label, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        # Add an overall title to the figure
        fig_lab.suptitle(
            f'Experiment "{experiment_name}": Sector Labor vs. {param_short_name}',
            fontsize=14,
        )
        # Adjust layout to prevent overlaps and accommodate the external legend
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])
        # Construct the full save path using the experiment subdirectory
        plot_path_lab = os.path.join(
            experiment_output_dir, f"{plot_filename_base}_Labor.png"
        )
        plt.savefig(plot_path_lab, bbox_inches="tight")  # Save the figure
        plt.close(fig_lab)  # Close the figure object to free memory
        print(f"Saved plot: {plot_path_lab}")

        # Plot: Intelligence Sector Wages (MPL_I)
        fig_mpl_i, ax_mpl_i = plt.subplots(figsize=(8, 4))
        for run_label, results_data in all_results.items():
            style = {"marker": ".", "markersize": 5}
            if run_label == "baseline":
                style = {
                    "lw": 2,
                    "color": "black",
                    "ls": "--",
                    "marker": "x",
                    "markersize": 4,
                }
            ax_mpl_i.plot(
                results_data["years"], results_data["MPL_I"], label=run_label, **style
            )
        ax_mpl_i.set_title(
            f'Experiment "{experiment_name}": $MPL_I$ vs. {param_short_name}'
        )
        ax_mpl_i.set_xlabel("Year")
        ax_mpl_i.set_ylabel("Wage ($MPL_I$)")
        ax_mpl_i.set_ylim(bottom=0)
        ax_mpl_i.legend(
            title=plot_param_label, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        ax_mpl_i.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_path_mpl_i = os.path.join(
            experiment_output_dir, f"{plot_filename_base}_MPL_I.png"
        )
        plt.savefig(plot_path_mpl_i, bbox_inches="tight")
        plt.close(fig_mpl_i)
        print(f"Saved plot: {plot_path_mpl_i}")

        # Plot: Traditional Sector Wages (MPL_T)
        fig_mpl_t, ax_mpl_t = plt.subplots(figsize=(8, 4))
        for run_label, results_data in all_results.items():
            style = {"marker": ".", "markersize": 5}
            if run_label == "baseline":
                style = {
                    "lw": 2,
                    "color": "black",
                    "ls": "--",
                    "marker": "x",
                    "markersize": 4,
                }
            ax_mpl_t.plot(
                results_data["years"], results_data["MPL_T"], label=run_label, **style
            )
        ax_mpl_t.set_title(
            f'Experiment "{experiment_name}": $MPL_T$ vs. {param_short_name}'
        )
        ax_mpl_t.set_xlabel("Year")
        ax_mpl_t.set_ylabel("Wage ($MPL_T$)")
        ax_mpl_t.set_ylim(bottom=0)
        ax_mpl_t.legend(
            title=plot_param_label, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        ax_mpl_t.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_path_mpl_t = os.path.join(
            experiment_output_dir, f"{plot_filename_base}_MPL_T.png"
        )
        plt.savefig(plot_path_mpl_t, bbox_inches="tight")
        plt.close(fig_mpl_t)
        print(f"Saved plot: {plot_path_mpl_t}")

        # Plot: Human Sector Wages (MPL_H)
        fig_mpl_h, ax_mpl_h = plt.subplots(figsize=(8, 4))
        for run_label, results_data in all_results.items():
            style = {"marker": ".", "markersize": 5}
            if run_label == "baseline":
                style = {
                    "lw": 2,
                    "color": "black",
                    "ls": "--",
                    "marker": "x",
                    "markersize": 4,
                }
            ax_mpl_h.plot(
                results_data["years"], results_data["MPL_H"], label=run_label, **style
            )
        ax_mpl_h.set_title(
            f'Experiment "{experiment_name}": $MPL_H$ vs. {param_short_name}'
        )
        ax_mpl_h.set_xlabel("Year")
        ax_mpl_h.set_ylabel("Wage ($MPL_H$)")
        ax_mpl_h.set_ylim(bottom=0)
        ax_mpl_h.legend(
            title=plot_param_label, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        ax_mpl_h.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_path_mpl_h = os.path.join(
            experiment_output_dir, f"{plot_filename_base}_MPL_H.png"
        )
        plt.savefig(plot_path_mpl_h, bbox_inches="tight")
        plt.close(fig_mpl_h)
        print(f"Saved plot: {plot_path_mpl_h}")

        # Plot: Total Output
        fig_ytot, ax_ytot = plt.subplots(figsize=(8, 4))
        for run_label, results_data in all_results.items():
            style = {"marker": ".", "markersize": 5}
            if run_label == "baseline":
                style = {
                    "lw": 2,
                    "color": "black",
                    "ls": "--",
                    "marker": "x",
                    "markersize": 4,
                }
            ax_ytot.plot(
                results_data["years"], results_data["Y_Total"], label=run_label, **style
            )
        ax_ytot.set_title(
            f'Experiment "{experiment_name}": Total Output vs. {param_short_name}'
        )
        ax_ytot.set_xlabel("Year")
        ax_ytot.set_ylabel("Total Output")
        ax_ytot.legend(
            title=plot_param_label, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        ax_ytot.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_path_ytot = os.path.join(
            experiment_output_dir, f"{plot_filename_base}_Y_Total.png"
        )
        plt.savefig(plot_path_ytot, bbox_inches="tight")
        plt.close(fig_ytot)
        print(f"Saved plot: {plot_path_ytot}")

        # Plot: Sector Outputs (Y_T, Y_H, Y_I) in separate panels
        fig_out, axes_out = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        for run_label, results_data in all_results.items():
            style = {"marker": ".", "markersize": 5}
            if run_label == "baseline":
                style = {
                    "lw": 2,
                    "color": "black",
                    "ls": "--",
                    "marker": "x",
                    "markersize": 4,
                }
            axes_out[0].plot(
                results_data["years"], results_data["Y_T"], label=run_label, **style
            )
            axes_out[1].plot(
                results_data["years"], results_data["Y_H"], label=run_label, **style
            )
            axes_out[2].plot(
                results_data["years"], results_data["Y_I"], label=run_label, **style
            )
        axes_out[0].set_title("Traditional Sector Output ($Y_T$)")
        axes_out[0].set_ylabel("Output ($Y_T$)")
        axes_out[0].grid(True)
        axes_out[1].set_title("Human Sector Output ($Y_H$)")
        axes_out[1].set_ylabel("Output ($Y_H$)")
        axes_out[1].grid(True)
        axes_out[2].set_title("Intelligence Sector Output ($Y_I$)")
        axes_out[2].set_ylabel("Output ($Y_I$)")
        axes_out[2].set_xlabel("Year")
        axes_out[2].grid(True)
        fig_out.legend(
            title=plot_param_label, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        fig_out.suptitle(
            f'Experiment "{experiment_name}": Sector Outputs vs. {param_short_name}',
            fontsize=14,
        )
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])
        plot_path_out = os.path.join(
            experiment_output_dir, f"{plot_filename_base}_SectorOutputs.png"
        )
        plt.savefig(plot_path_out, bbox_inches="tight")
        plt.close(fig_out)
        print(f"Saved plot: {plot_path_out}")

        # Plot: Traditional Sector MPK_T
        fig_mpk_t, ax_mpk_t = plt.subplots(figsize=(8, 4))
        for run_label, results_data in all_results.items():
            style = {"marker": ".", "markersize": 5}
            if run_label == "baseline":
                style = {
                    "lw": 2,
                    "color": "black",
                    "ls": "--",
                    "marker": "x",
                    "markersize": 4,
                }
            ax_mpk_t.plot(
                results_data["years"], results_data["MPK_T"], label=run_label, **style
            )
        ax_mpk_t.set_title(
            f'Experiment "{experiment_name}": $MPK_T$ vs. {param_short_name}'
        )
        ax_mpk_t.set_xlabel("Year")
        ax_mpk_t.set_ylabel("MPK_T")
        ax_mpk_t.set_ylim(bottom=0)
        ax_mpk_t.legend(
            title=plot_param_label, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        ax_mpk_t.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_path_mpk_t = os.path.join(
            experiment_output_dir, f"{plot_filename_base}_MPK_T.png"
        )
        plt.savefig(plot_path_mpk_t, bbox_inches="tight")
        plt.close(fig_mpk_t)
        print(f"Saved plot: {plot_path_mpk_t}")

        # Plot: Traditional Sector MPK_I
        fig_mpk_i, ax_mpk_i = plt.subplots(figsize=(8, 4))
        for run_label, results_data in all_results.items():
            style = {"marker": ".", "markersize": 5}
            if run_label == "baseline":
                style = {
                    "lw": 2,
                    "color": "black",
                    "ls": "--",
                    "marker": "x",
                    "markersize": 4,
                }
            ax_mpk_i.plot(
                results_data["years"], results_data["MPK_I"], label=run_label, **style
            )
        ax_mpk_i.set_title(
            f'Experiment "{experiment_name}": $MPK_I$ vs. {param_short_name}'
        )
        ax_mpk_i.set_xlabel("Year")
        ax_mpk_i.set_ylabel("MPK_I")
        ax_mpk_i.set_ylim(bottom=0)
        ax_mpk_i.legend(
            title=plot_param_label, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        ax_mpk_i.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_path_mpk_i = os.path.join(
            experiment_output_dir, f"{plot_filename_base}_MPK_I.png"
        )
        plt.savefig(plot_path_mpk_i, bbox_inches="tight")
        plt.close(fig_mpk_i)
        print(f"Saved plot: {plot_path_mpk_i}")

        # Plot: Traditional Sector MPK_H
        fig_mpk_h, ax_mpk_h = plt.subplots(figsize=(8, 4))
        for run_label, results_data in all_results.items():
            style = {"marker": ".", "markersize": 5}
            if run_label == "baseline":
                style = {
                    "lw": 2,
                    "color": "black",
                    "ls": "--",
                    "marker": "x",
                    "markersize": 4,
                }
            ax_mpk_h.plot(
                results_data["years"], results_data["MPK_H"], label=run_label, **style
            )
        ax_mpk_h.set_title(
            f'Experiment "{experiment_name}": $MPK_H$ vs. {param_short_name}'
        )
        ax_mpk_h.set_xlabel("Year")
        ax_mpk_h.set_ylabel("MPK_H")
        ax_mpk_h.set_ylim(bottom=0)
        ax_mpk_h.legend(
            title=plot_param_label, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        ax_mpk_h.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_path_mpk_h = os.path.join(
            experiment_output_dir, f"{plot_filename_base}_MPK_H.png"
        )
        plt.savefig(plot_path_mpk_h, bbox_inches="tight")
        plt.close(fig_mpk_h)
        print(f"Saved plot: {plot_path_mpk_h}")

        # Plot: Traditional Sector MPA_T
        fig_mpa_t, ax_mpa_t = plt.subplots(figsize=(8, 4))
        for run_label, results_data in all_results.items():
            style = {"marker": ".", "markersize": 5}
            if run_label == "baseline":
                style = {
                    "lw": 2,
                    "color": "black",
                    "ls": "--",
                    "marker": "x",
                    "markersize": 4,
                }
            ax_mpa_t.plot(
                results_data["years"], results_data["MPA_T"], label=run_label, **style
            )
        ax_mpa_t.set_title(
            f'Experiment "{experiment_name}": $MPA_T$ vs. {param_short_name}'
        )
        ax_mpa_t.set_xlabel("Year")
        ax_mpa_t.set_ylabel("MPA_T")
        ax_mpa_t.set_ylim(bottom=0)
        ax_mpa_t.legend(
            title=plot_param_label, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        ax_mpa_t.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_path_mpa_t = os.path.join(
            experiment_output_dir, f"{plot_filename_base}_MPA_T.png"
        )
        plt.savefig(plot_path_mpa_t, bbox_inches="tight")
        plt.close(fig_mpa_t)
        print(f"Saved plot: {plot_path_mpa_t}")

        # Plot: Traditional Sector MPA_I
        fig_mpa_i, ax_mpa_i = plt.subplots(figsize=(8, 4))
        for run_label, results_data in all_results.items():
            style = {"marker": ".", "markersize": 5}
            if run_label == "baseline":
                style = {
                    "lw": 2,
                    "color": "black",
                    "ls": "--",
                    "marker": "x",
                    "markersize": 4,
                }
            ax_mpa_i.plot(
                results_data["years"], results_data["MPA_I"], label=run_label, **style
            )
        ax_mpa_i.set_title(
            f'Experiment "{experiment_name}": $MPA_I$ vs. {param_short_name}'
        )
        ax_mpa_i.set_xlabel("Year")
        ax_mpa_i.set_ylabel("MPA_I")
        ax_mpa_i.set_ylim(bottom=0)
        ax_mpa_i.legend(
            title=plot_param_label, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        ax_mpa_i.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_path_mpa_i = os.path.join(
            experiment_output_dir, f"{plot_filename_base}_MPA_I.png"
        )
        plt.savefig(plot_path_mpa_i, bbox_inches="tight")
        plt.close(fig_mpa_i)
        print(f"Saved plot: {plot_path_mpa_i}")

        # Plot: Unemployment Rate
        fig_unemployment, ax_unemployment = plt.subplots(figsize=(8, 4))
        for run_label, results_data in all_results.items():
            style = {"marker": ".", "markersize": 5}
            if run_label == "baseline":
                style = {
                    "lw": 2,
                    "color": "black",
                    "ls": "--",
                    "marker": "x",
                    "markersize": 4,
                }
            ax_unemployment.plot(
                results_data["years"], results_data["L_U"], label=run_label, **style
            )
        ax_unemployment.set_title(
            f'Experiment "{experiment_name}": Unemployment Rate vs. {param_short_name}'
        )
        ax_unemployment.set_xlabel("Year")
        ax_unemployment.set_ylabel("Unemployment Rate")
        ax_unemployment.set_ylim(bottom=0)
        ax_unemployment.legend(
            title=plot_param_label, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        ax_unemployment.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_path_unemployment = os.path.join(
            experiment_output_dir, f"{plot_filename_base}_U.png"
        )
        plt.savefig(plot_path_unemployment, bbox_inches="tight")
        plt.close(fig_unemployment)
        print(f"Saved plot: {plot_path_unemployment}")

        print(f"\nExperiment '{experiment_name}' Run and Plotting Complete.")

    # --- Save Results Dictionary ---
    # Save the entire 'all_results' dictionary (including baseline) to a pickle file
    # for potential later analysis without re-running the simulations.
    pickle_filename = os.path.join(
        experiment_output_dir, f"{plot_filename_base}_results.pkl"
    )
    try:
        with open(pickle_filename, "wb") as f:
            pickle.dump(all_results, f)  # Use pickle to serialize the dictionary
        print(f"Saved detailed results to {pickle_filename}")
    except Exception as e:
        # Handle potential errors during file saving
        print(f"Error saving results to pickle file '{pickle_filename}': {e}")

    # Return the results dictionary containing data from all runs
    return all_results


def run_reallocation_experiment(
    variable_type,
    target_sector,
    reallocation_fractions,
    experiment_name="default_reallocation",
    do_plots=True,
):
    """
    Runs a simulation experiment by reallocating a variable (K, L, or A) from other sectors
    to a target sector while keeping the total constant. This allows studying the effects
    of resource reallocation rather than absolute changes.

    Args:
        variable_type (str): Type of variable to reallocate ('K', 'L', or 'A')
        target_sector (str): Sector to increase ('T', 'H', or 'I')
        reallocation_fractions (list or np.array): Fractions of total to allocate to target sector
                                                   (e.g., [0.3, 0.4, 0.5] means target gets 30%, 40%, 50% of total)
        experiment_name (str): Descriptive name for this experiment
        do_plots (bool): Whether to generate plots

    Returns:
        dict: Dictionary containing simulation results for baseline and all reallocation scenarios
    """
    # --- Define Main Output Directory ---
    main_output_dir = "../results/experiments"

    # --- Input Validation ---
    if variable_type not in ["K", "L", "A"]:
        print("Error: variable_type must be 'K', 'L', or 'A'")
        return None

    if target_sector not in ["T", "H", "I"]:
        print("Error: target_sector must be 'T', 'H', or 'I'")
        return None

    if (
        not isinstance(reallocation_fractions, Iterable)
        or isinstance(reallocation_fractions, str)
        or len(reallocation_fractions) == 0
    ):
        print("Error: reallocation_fractions must be a non-empty list or numpy array.")
        return None

    if not isinstance(experiment_name, str) or not experiment_name:
        print("Error: experiment_name must be a non-empty string.")
        return None

    # Sanitize experiment name
    safe_experiment_name = re.sub(r"[^\w\-_\. ]", "_", experiment_name)
    if safe_experiment_name != experiment_name:
        print(
            f"Warning: Experiment name sanitized to '{safe_experiment_name}' for directory creation."
        )
        experiment_name = safe_experiment_name

    experiment_output_dir = os.path.join(main_output_dir, experiment_name)

    # --- Create Output Directories ---
    try:
        os.makedirs(main_output_dir, exist_ok=True)
        os.makedirs(experiment_output_dir, exist_ok=True)
        print(f"Ensured output directory exists: '{experiment_output_dir}'")
    except OSError as e:
        print(f"Error creating output directories: {e}")
        return None

    # --- Print Experiment Setup ---
    print("\n" + "=" * 60)
    print(f"Starting Reallocation Experiment: '{experiment_name}'")
    print(f"Reallocating Variable: '{variable_type}' to sector '{target_sector}'")
    print(f"Target sector fractions to test: {reallocation_fractions}")
    print(f"Output will be saved in: '{experiment_output_dir}'")
    print("=" * 60)

    # --- Load Fixed Parameters & Generate Dependent Vars from Config ---
    T_sim = copy.deepcopy(config.T_sim)
    L_total = copy.deepcopy(config.L_total)
    years = np.arange(T_sim + 1)

    # Generate AI adoption curves
    phi_T_t = config.phi_T_init + (config.phi_T_max - config.phi_T_init) / (
        1 + np.exp(-config.phi_T_k * (years - config.phi_T_t0))
    )
    phi_T_t[0] = config.phi_T_init
    phi_I_t = config.phi_I_init + (config.phi_I_max - config.phi_I_init) / (
        1 + np.exp(-config.phi_I_k * (years - config.phi_I_t0))
    )
    phi_I_t[0] = config.phi_I_init

    # Load parameter dictionaries
    initial_conditions = copy.deepcopy(config.initial_conditions)
    economic_params = copy.deepcopy(config.economic_params)
    base_production_params = copy.deepcopy(config.base_production_params)
    labor_mobility_params = copy.deepcopy(config.labor_mobility_params)

    # --- Get baseline totals for the variable type ---
    if variable_type == "K":
        baseline_total = (
            initial_conditions["K_T_0"]
            + initial_conditions["K_H_0"]
            + initial_conditions["K_I_0"]
        )
        var_keys = ["K_T_0", "K_H_0", "K_I_0"]
        baseline_values = [initial_conditions[key] for key in var_keys]
    elif variable_type == "L":
        # For labor, we only reallocate employed labor (not unemployment)
        baseline_total = (
            initial_conditions["L_T_0"]
            + initial_conditions["L_H_0"]
            + initial_conditions["L_I_0"]
        )
        var_keys = ["L_T_0", "L_H_0", "L_I_0"]
        baseline_values = [initial_conditions[key] for key in var_keys]
    elif variable_type == "A":
        baseline_total = (
            initial_conditions["A_T_0"]
            + initial_conditions["A_H_0"]
            + initial_conditions["A_I_0"]
        )
        var_keys = ["A_T_0", "A_H_0", "A_I_0"]
        baseline_values = [initial_conditions[key] for key in var_keys]

    print(f"Baseline {variable_type} total: {baseline_total:.2f}")
    print(
        f"Baseline {variable_type} distribution: T={baseline_values[0]:.2f}, H={baseline_values[1]:.2f}, I={baseline_values[2]:.2f}"
    )
    print("-" * 60)

    # --- Run Baseline Simulation ---
    print("Running Baseline Simulation (using original config values)...")
    baseline_start_time = time.time()
    baseline_results = run_single_simulation(
        T_sim=T_sim,
        L_total=L_total,
        initial_conditions=config.initial_conditions,
        economic_params=config.economic_params,
        production_params=config.base_production_params,
        labor_mobility_params=config.labor_mobility_params,
        phi_T_t=phi_T_t,
        phi_I_t=phi_I_t,
        verbose=False,
    )
    baseline_end_time = time.time()
    print(
        f"Baseline simulation finished in {baseline_end_time - baseline_start_time:.2f} seconds."
    )
    print("--- Baseline Final State ---")
    print(f"  Total Output: {baseline_results['Y_Total'][-1]:.2f}")
    print(
        f"  Labor (T, H, I, U): {baseline_results['L_T'][-1]:.1f}, {baseline_results['L_H'][-1]:.1f}, {baseline_results['L_I'][-1]:.1f}, {baseline_results['L_U'][-1]:.1f}"
    )
    print(
        f"  Wages (T, H, I): {baseline_results['MPL_T'][-1]:.3f}, {baseline_results['MPL_H'][-1]:.3f}, {baseline_results['MPL_I'][-1]:.3f}"
    )
    print("-" * 28)

    # --- Experiment Execution ---
    print("\nStarting Reallocation Simulations...")
    all_results = {}
    all_results["baseline"] = baseline_results
    exp_start_time = time.time()

    # Map target sector to index
    sector_indices = {"T": 0, "H": 1, "I": 2}
    target_idx = sector_indices[target_sector]
    other_indices = [i for i in range(3) if i != target_idx]

    for i, target_fraction in enumerate(reallocation_fractions):
        # Create run label
        run_label = f"{variable_type}_{target_sector}_frac={target_fraction:.3f}"
        print(
            f"\nRunning Simulation {i+1}/{len(reallocation_fractions)}: {run_label}..."
        )

        # --- Create Deep Copies of Parameter Dictionaries for This Run ---
        current_initial_conditions = copy.deepcopy(initial_conditions)
        current_economic_params = copy.deepcopy(economic_params)
        current_production_params = copy.deepcopy(base_production_params)
        current_labor_mobility_params = copy.deepcopy(labor_mobility_params)

        # --- Calculate new allocation ---
        new_values = [0.0, 0.0, 0.0]

        # Set target sector value
        new_values[target_idx] = baseline_total * target_fraction

        # Distribute remaining among other sectors proportionally to their baseline shares
        remaining_total = baseline_total * (1 - target_fraction)
        other_baseline_total = sum(baseline_values[j] for j in other_indices)

        if other_baseline_total > 0:  # Avoid division by zero
            for j in other_indices:
                proportion = baseline_values[j] / other_baseline_total
                new_values[j] = remaining_total * proportion
        else:
            # If other sectors had zero baseline, distribute equally
            for j in other_indices:
                new_values[j] = remaining_total / len(other_indices)

        # --- Apply new values to initial conditions ---
        for j, key in enumerate(var_keys):
            current_initial_conditions[key] = new_values[j]

        print(
            f"  New {variable_type} allocation: T={new_values[0]:.2f}, H={new_values[1]:.2f}, I={new_values[2]:.2f}"
        )
        print(f"  Total: {sum(new_values):.2f} (should equal {baseline_total:.2f})")

        # --- Run the Simulation ---
        try:
            simulation_output = run_single_simulation(
                T_sim=T_sim,
                L_total=L_total,
                initial_conditions=current_initial_conditions,
                economic_params=current_economic_params,
                production_params=current_production_params,
                labor_mobility_params=current_labor_mobility_params,
                phi_T_t=phi_T_t,
                phi_I_t=phi_I_t,
                verbose=False,
            )
            all_results[run_label] = simulation_output
            print(
                f"Finished run {run_label}. Final Total Output: {simulation_output['Y_Total'][-1]:.2f}"
            )
        except Exception as e:
            print(f"  ERROR during simulation run for {run_label}: {e}. Skipping.")
            continue

    # --- Timing and Completion Message ---
    exp_end_time = time.time()
    print("-" * 60)
    print(
        f"All reallocation simulations completed in {exp_end_time - exp_start_time:.2f} seconds."
    )
    print("-" * 60)

    if do_plots:
        ####################################
        # --- Analysis and Visualization ---
        ####################################
        if not all_results or len(all_results) <= 1:
            print("No successful reallocation simulations to plot.")
            return all_results

        print(f"Generating comparison plots in '{experiment_output_dir}'...")
        plt.style.use("seaborn-v0_8-darkgrid")
        plot_param_label = f"{variable_type} to {target_sector} fraction"
        plot_filename_base = f"realloc_{variable_type}_to_{target_sector}"

        # --- Plot Generation (same plots as regular experiments) ---

        # Plot: Sector Labor Allocations
        fig_lab, axes_lab = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        for run_label, results_data in all_results.items():
            style = {"marker": ".", "markersize": 5}
            if run_label == "baseline":
                style = {
                    "lw": 2,
                    "color": "black",
                    "ls": "--",
                    "marker": "x",
                    "markersize": 4,
                }
            axes_lab[0].plot(
                results_data["years"], results_data["L_I"], label=run_label, **style
            )
            axes_lab[1].plot(
                results_data["years"], results_data["L_H"], label=run_label, **style
            )
            axes_lab[2].plot(
                results_data["years"], results_data["L_T"], label=run_label, **style
            )

        axes_lab[0].set_title(f"Intelligence Sector Labor ($L_I$)")
        axes_lab[0].set_ylabel("Labor Units ($L_I$)")
        axes_lab[0].grid(True)
        axes_lab[1].set_title(f"Human Sector Labor ($L_H$)")
        axes_lab[1].set_ylabel("Labor Units ($L_H$)")
        axes_lab[1].grid(True)
        axes_lab[2].set_title(f"Traditional Sector Labor ($L_T$)")
        axes_lab[2].set_ylabel("Labor Units ($L_T$)")
        axes_lab[2].set_xlabel("Year")
        axes_lab[2].grid(True)

        fig_lab.legend(
            title=plot_param_label, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        fig_lab.suptitle(
            f'Reallocation Experiment "{experiment_name}": Sector Labor vs. {variable_type} to {target_sector}',
            fontsize=14,
        )
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])
        plot_path_lab = os.path.join(
            experiment_output_dir, f"{plot_filename_base}_Labor.png"
        )
        plt.savefig(plot_path_lab, bbox_inches="tight")
        plt.close(fig_lab)
        print(f"Saved plot: {plot_path_lab}")

        # Plot: Total Output
        fig_ytot, ax_ytot = plt.subplots(figsize=(8, 4))
        for run_label, results_data in all_results.items():
            style = {"marker": ".", "markersize": 5}
            if run_label == "baseline":
                style = {
                    "lw": 2,
                    "color": "black",
                    "ls": "--",
                    "marker": "x",
                    "markersize": 4,
                }
            ax_ytot.plot(
                results_data["years"], results_data["Y_Total"], label=run_label, **style
            )

        ax_ytot.set_title(
            f'Reallocation Experiment "{experiment_name}": Total Output vs. {variable_type} to {target_sector}'
        )
        ax_ytot.set_xlabel("Year")
        ax_ytot.set_ylabel("Total Output")
        ax_ytot.legend(
            title=plot_param_label, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        ax_ytot.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_path_ytot = os.path.join(
            experiment_output_dir, f"{plot_filename_base}_Y_Total.png"
        )
        plt.savefig(plot_path_ytot, bbox_inches="tight")
        plt.close(fig_ytot)
        print(f"Saved plot: {plot_path_ytot}")

        # Plot: Sector Wages
        fig_wages, axes_wages = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        for run_label, results_data in all_results.items():
            style = {"marker": ".", "markersize": 5}
            if run_label == "baseline":
                style = {
                    "lw": 2,
                    "color": "black",
                    "ls": "--",
                    "marker": "x",
                    "markersize": 4,
                }
            axes_wages[0].plot(
                results_data["years"], results_data["MPL_T"], label=run_label, **style
            )
            axes_wages[1].plot(
                results_data["years"], results_data["MPL_H"], label=run_label, **style
            )
            axes_wages[2].plot(
                results_data["years"], results_data["MPL_I"], label=run_label, **style
            )

        axes_wages[0].set_title("Traditional Sector Wages ($MPL_T$)")
        axes_wages[0].set_ylabel("Wage ($MPL_T$)")
        axes_wages[0].grid(True)
        axes_wages[1].set_title("Human Sector Wages ($MPL_H$)")
        axes_wages[1].set_ylabel("Wage ($MPL_H$)")
        axes_wages[1].grid(True)
        axes_wages[2].set_title("Intelligence Sector Wages ($MPL_I$)")
        axes_wages[2].set_ylabel("Wage ($MPL_I$)")
        axes_wages[2].set_xlabel("Year")
        axes_wages[2].grid(True)

        fig_wages.legend(
            title=plot_param_label, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        fig_wages.suptitle(
            f'Reallocation Experiment "{experiment_name}": Sector Wages vs. {variable_type} to {target_sector}',
            fontsize=14,
        )
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])
        plot_path_wages = os.path.join(
            experiment_output_dir, f"{plot_filename_base}_Wages.png"
        )
        plt.savefig(plot_path_wages, bbox_inches="tight")
        plt.close(fig_wages)
        print(f"Saved plot: {plot_path_wages}")

        print(
            f"\nReallocation Experiment '{experiment_name}' Run and Plotting Complete."
        )

    # --- Save Results Dictionary ---
    pickle_filename = os.path.join(
        experiment_output_dir, f"{plot_filename_base}_results.pkl"
    )
    try:
        with open(pickle_filename, "wb") as f:
            pickle.dump(all_results, f)
        print(f"Saved detailed results to {pickle_filename}")
    except Exception as e:
        print(f"Error saving results to pickle file '{pickle_filename}': {e}")

    return all_results


##############################
# --- Main Execution Block ---
##############################
if __name__ == "__main__":
    # This block executes only when the script is run directly (not imported as a module).
    # Define and run the desired parameter experiments here.

    # Print out baseline parameters
    print(config.base_production_params)
    print(config.initial_conditions)
    print(config.economic_params)
    print(config.labor_mobility_params)

    # Experiments
    sector_T_production_params = True
    sector_I_production_params = True
    sector_H_production_params = True

    # Reallocation experiments
    sector_T_initial_conditions = True
    sector_I_initial_conditions = True
    sector_H_initial_conditions = True

    AI_adoption_params = True  # Set to False to focus on reallocation experiments

    economic_params = True  # Set to False to focus on reallocation experiments
    labor_mobility_params = True

    # Traditional Sector Production Function Parameters
    if sector_T_production_params:
        # === Experiment: Varying alpha in Sector T ===
        run_parameter_experiment(
            param_path_str="base_production_params['T']['alpha']",
            param_values=np.linspace(0.1, 1.0, 10),
            experiment_name="vary_alpha_T",
        )
        # === Experiment: Varying rho_outer in Sector T ===
        run_parameter_experiment(
            param_path_str="base_production_params['T']['rho_outer']",
            param_values=np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1]),
            experiment_name="vary_rho_T_outer",
        )
        # === Experiment: Varying rho_inner in Sector T ===
        run_parameter_experiment(
            param_path_str="base_production_params['T']['rho_inner']",
            param_values=np.array([0.4, 0.5, 0.6, 0.7]),
            experiment_name="vary_rho_T_inner",
        )

    # Intelligence Sector Production Function Parameters
    if sector_I_production_params:
        # === Experiment: Varying alpha in Sector I ===
        run_parameter_experiment(
            param_path_str="base_production_params['I']['alpha']",
            param_values=np.linspace(0.1, 1.0, 10),
            experiment_name="vary_alpha_I",
        )
        # === Experiment: Varying rho_outer in Sector I ===
        run_parameter_experiment(
            param_path_str="base_production_params['I']['rho_outer']",
            param_values=np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]),
            experiment_name="vary_rho_I_outer",
        )
        # === Experiment: Varying rho_inner in Sector I ===
        run_parameter_experiment(
            param_path_str="base_production_params['I']['rho_inner']",
            param_values=np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            experiment_name="vary_rho_I_inner",
        )

    # Human Sector Production Function Parameters
    if sector_H_production_params:
        # === Experiment: Varying alpha in Sector H ===
        run_parameter_experiment(
            param_path_str="base_production_params['H']['alpha']",
            param_values=np.linspace(0.1, 1.0, 10),
            experiment_name="vary_alpha_H",
        )
        # === Experiment: Varying rho_outer in Sector H ===
        run_parameter_experiment(
            param_path_str="base_production_params['H']['rho_outer']",
            param_values=np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]),
            experiment_name="vary_rho_H_outer",
        )

    # === REALLOCATION EXPERIMENTS ===
    # These experiments reallocate resources between sectors while keeping totals constant

    # Capital (K) Reallocation Experiments
    if sector_T_initial_conditions:
        # Reallocate capital TO Traditional sector
        run_reallocation_experiment(
            variable_type="K",
            target_sector="T",
            reallocation_fractions=np.array([0.3, 0.4, 0.5, 0.6, 0.7]),
            experiment_name="realloc_K_to_T",
        )

    if sector_I_initial_conditions:
        # Reallocate capital TO Intelligence sector
        run_reallocation_experiment(
            variable_type="K",
            target_sector="I",
            reallocation_fractions=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            experiment_name="realloc_K_to_I",
        )

    if sector_H_initial_conditions:
        # Reallocate capital TO Human sector
        run_reallocation_experiment(
            variable_type="K",
            target_sector="H",
            reallocation_fractions=np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
            experiment_name="realloc_K_to_H",
        )

    # Labor (L) Reallocation Experiments
    if sector_T_initial_conditions:
        # Reallocate labor TO Traditional sector
        run_reallocation_experiment(
            variable_type="L",
            target_sector="T",
            reallocation_fractions=np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
            experiment_name="realloc_L_to_T",
        )

    if sector_I_initial_conditions:
        # Reallocate labor TO Intelligence sector
        run_reallocation_experiment(
            variable_type="L",
            target_sector="I",
            reallocation_fractions=np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
            experiment_name="realloc_L_to_I",
        )

    if sector_H_initial_conditions:
        # Reallocate labor TO Human sector
        run_reallocation_experiment(
            variable_type="L",
            target_sector="H",
            reallocation_fractions=np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
            experiment_name="realloc_L_to_H",
        )

    # AI Capital (A) Reallocation Experiments
    if sector_T_initial_conditions:
        # Reallocate AI capital TO Traditional sector
        run_reallocation_experiment(
            variable_type="A",
            target_sector="T",
            reallocation_fractions=np.array([0.4, 0.5, 0.6, 0.7, 0.8]),
            experiment_name="realloc_A_to_T",
        )

    if sector_I_initial_conditions:
        # Reallocate AI capital TO Intelligence sector
        run_reallocation_experiment(
            variable_type="A",
            target_sector="I",
            reallocation_fractions=np.array([0.3, 0.4, 0.5, 0.6, 0.7]),
            experiment_name="realloc_A_to_I",
        )

        # Note: We don't reallocate A to H sector since H has no AI in the baseline model

    # AI Adoption Parameters
    if AI_adoption_params:
        # === Experiment: Varying phi_T_max ===
        run_parameter_experiment(
            param_path_str="phi_T_max",
            param_values=np.array([0.5, 0.7, 0.9, 0.95, 0.99]),
            experiment_name="vary_phi_T_max",
        )
        # === Experiment: Varying phi_I_max ===
        run_parameter_experiment(
            param_path_str="phi_I_max",
            param_values=np.array([0.5, 0.7, 0.9, 0.95, 0.99]),
            experiment_name="vary_phi_I_max",
        )
        # === Experiment: Varying phi_T_k (adoption speed) ===
        run_parameter_experiment(
            param_path_str="phi_T_k",
            param_values=np.array([0.3, 0.5, 0.8, 1.0, 1.5]),
            experiment_name="vary_phi_T_k",
        )
        # === Experiment: Varying phi_I_k (adoption speed) ===
        run_parameter_experiment(
            param_path_str="phi_I_k",
            param_values=np.array([0.3, 0.5, 0.8, 1.0, 1.5]),
            experiment_name="vary_phi_I_k",
        )

    # Economic Parameters
    if economic_params:
        # === Experiment: Varying Aggregate Savings Rate for K ===
        run_parameter_experiment(
            param_path_str="economic_params['s_K']",
            param_values=np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
            experiment_name="vary_s_K",
        )
        # === Experiment: Varying Aggregate Savings Rate for AI ===
        run_parameter_experiment(
            param_path_str="economic_params['s_A']",
            param_values=np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
            experiment_name="vary_s_A",
        )
        # === Experiment: Varying depreciation rate for K ===
        run_parameter_experiment(
            param_path_str="economic_params['delta_K']",
            param_values=np.array([0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]),
            experiment_name="vary_delta_K",
        )
        # === Experiment: Varying depreciation rate for AI ===
        run_parameter_experiment(
            param_path_str="economic_params['delta_A']",
            param_values=np.array([0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]),
            experiment_name="vary_delta_A",
        )
        # === Experiment: Varying capital sensitivity ===
        run_parameter_experiment(
            param_path_str="economic_params['capital_sensitivity']",
            param_values=np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]),
            experiment_name="vary_capital_sensitivity",
        )

    # Labor Mobility Parameters
    if labor_mobility_params:
        # === Experiment: Varying Labor Mobility Factor ===
        run_parameter_experiment(
            param_path_str="labor_mobility_params['mobility_factor']",
            param_values=np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0]),
            experiment_name="vary_mobility_factor",
        )
        # === Experiment: Varying Non-Movable Fraction ===
        run_parameter_experiment(
            param_path_str="labor_mobility_params['non_movable_fraction']",
            param_values=np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0]),
            experiment_name="vary_non_movable_fraction",
        )
        # === Experiment: Varying Job Finding Rate ===
        run_parameter_experiment(
            param_path_str="labor_mobility_params['job_finding_rate']",
            param_values=np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0]),
            experiment_name="vary_job_finding_rate",
        )
        # === Experiment: Varying Job Separation Rate ===
        run_parameter_experiment(
            param_path_str="labor_mobility_params['job_separation_rate']",
            param_values=np.array([0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]),
            experiment_name="vary_job_separation_rate",
        )
        # === Experiment: Varying Wage Sensitivity ===
        run_parameter_experiment(
            param_path_str="labor_mobility_params['wage_sensitivity']",
            param_values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            experiment_name="vary_wage_sensitivity",
        )

    print("\n\n=== ALL EXPERIMENTS FINISHED ===")
    # Optional: Access returned results dictionaries here if needed for further processing
    # e.g., if results_rho_I: print(...)
