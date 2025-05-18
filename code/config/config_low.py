#!/usr/bin/env python
# Low variant configuration for Norway
# Inherits from base_norway and allows parameter adjustments

# Import base configuration
from config.config_base import *

# Dictionary to store parameter adjustments
param_adjustments = {}


def set_parameter(param_path, value):
    """
    Set a specific parameter value, overriding the base configuration.

    Args:
        param_path (str): Dot-notation path to parameter (e.g., 'economic_params.s_A')
        value: New value to set
    """
    param_adjustments[param_path] = value

    # Apply the change immediately
    path_parts = param_path.split(".")

    if len(path_parts) == 1:
        # Top-level parameter
        globals()[path_parts[0]] = value
    elif len(path_parts) == 2:
        # Nested parameter in dictionary
        container, param = path_parts
        if container in globals():
            globals()[container][param] = value
    elif len(path_parts) == 3:
        # Doubly nested parameter (e.g., base_production_params.T.alpha)
        container, subcontainer, param = path_parts
        if container in globals() and subcontainer in globals()[container]:
            globals()[container][subcontainer][param] = value


def get_adjusted_parameters():
    """Return dictionary of all parameters that differ from base config"""
    return param_adjustments


# Parameter adjustments for low scenario:
set_parameter("s_A", 0.05)  # Lower AI investment rate
# set_parameter("phi_I_t0", 12)  # Later inflection point for AI adoption in Intelligence sector
# set_parameter("phi_T_max", 0.70)  # Lower maximum AI share in Traditional sector
# set_parameter("phi_I_max", 0.80)  # Lower maximum AI share in Intelligence sector
# set_parameter("phi_I_k", 0.30)  # Slower adoption speed in Intelligence sector
# set_parameter("phi_T_k", 0.20)  # Slower adoption speed in Traditional sector
# set_parameter("economic_params.s_A", 0.10)  # Lower AI investment rate
