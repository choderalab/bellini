# =============================================================================
# IMPORTS
# =============================================================================
import bellini as be
import pint
ureg = pint.UnitRegistry()

# =============================================================================
# PRE-DEFINED GROUPS
# =============================================================================
def substance():
    # initialize substance
    substance = be.group.Group()

    # specify allowed units
    substance.allowed_units = [ureg.mole, ureg.mol/ureg.liter]

    return substance

def well():
    # initialize well
    well = be.group.Group()

    # specify allowed units
    well.allowed_units = [ureg.liter]
    well.allow_children = True

    return well

def plate(
        number_of_wells=96,
    ):
    # initialize plate
    plate = be.group.Group()

    plate.allow_children = True
    plate.allowed_units = []

    # put wells in plate
    for idx in range(number_of_wells):
        plate['well%s' % idx] = well()

    return plate
