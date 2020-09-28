# =============================================================================
# IMPORTS
# =============================================================================
import abc
import bellini as be
import pint; ureg = pint.UnitRegistry()

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Substance(abc.ABC):
    """ Base class for substance with species and quantity. """
    def __init__(self, species, mole, **extras):
        # assert the type for species and mole
        assert isinstance(
            species,
            be.species.Species
        )

        assert isinstance(
            mole,
            be.quantity.Quantity,
        )

        # assert the unit of mole is indeed mole
        assert mole.unit == ureg.mole

        self.species = species
        self.mole = mole
        self.extras = extras
