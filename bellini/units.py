# =============================================================================
# IMPORTS
# =============================================================================
import warnings
import pint
ureg = pint.UnitRegistry()

# =============================================================================
# CONSTANTS
# =============================================================================
QUANTITY_UNIT = ureg.mole
VOLUME_UNIT = ureg.liter
MASS_UNIT = ureg.gram
CONCENTRATION_UNIT = ureg.mole / ureg.liter

UNITS = [
    QUANTITY_UNIT,
    VOLUME_UNIT,
    MASS_UNIT,
    CONCENTRATION_UNIT,
]

def get_internal_units(q):
    dim = q.dimensionality
    for unit in UNITS:
        if dim == unit.dimensionality:
            return unit
    return q.to_base_units().units

def to_internal_units(q):
    internal_unit = get_internal_units(q)
    return q.to_units(internal_unit)
