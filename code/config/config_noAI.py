#!/usr/bin/env python
# Zero AI adoption configuration for Norway
# Inherits from base_norway and sets AI-related parameters to zero

from config.config_base import *

# Dictionary to keep track of changes relative to the base config
param_adjustments = {}


def set_parameter(param_path, value):
    """Override a parameter from the base configuration."""
    param_adjustments[param_path] = value

    path_parts = param_path.split(".")

    if len(path_parts) == 1:
        globals()[path_parts[0]] = value
    elif len(path_parts) == 2:
        container, param = path_parts
        if container in globals():
            globals()[container][param] = value
    elif len(path_parts) == 3:
        container, subcontainer, param = path_parts
        if container in globals() and subcontainer in globals()[container]:
            globals()[container][subcontainer][param] = value


def get_adjusted_parameters():
    """Return dictionary of all parameters overridden in this config."""
    return param_adjustments


# --- Disable AI adoption entirely ---
set_parameter("phi_T_max", 0.05)
set_parameter("phi_T_init", 0.05)
set_parameter("phi_I_max", 0.15)
set_parameter("phi_I_init", 0.15)
