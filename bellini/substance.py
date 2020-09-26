# =============================================================================
# IMPORTS
# =============================================================================
import abc
import bellini as be

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Substance(abc.ABC):
    """ Base class for all substances. """
    def __init__(self, mole, **extras):
        self.mole = mole
        self.extras = extras


class Solvent(Substance):
    """ Solvent. """
    def __init__(self, volume, molar_volume, **extras):

        # compute the quantity
        mole = volume / molar_volume

        # initialize a substance class
        super(Solvent, self).__init__(
            mol=mole,
            **extras,
        )

        self.volume = volume
        self.molar_volume = molar_volume
