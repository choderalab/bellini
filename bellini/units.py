# =============================================================================
# IMPORTS
# =============================================================================
import pint
ureg = pint.UnitRegistry()

# =============================================================================
# CONSTANTS
# =============================================================================
QUANTITY_UNIT = ureg.mole
VOLUME_UNIT = ureg.milliliter
MASS_UNIT = ureg.gram


SYSTEM_UNITS = [value for key, value in vars().items() if key.endswith("_UNIT")]
