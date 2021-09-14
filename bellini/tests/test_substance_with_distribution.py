import pytest
from bellini.units import *

def test_init():
    import bellini
    from bellini import Substance, Quantity, Species
    from bellini.distributions import Normal

    bellini.verbose = True

    quantity_distribution = Normal(
        loc=Quantity(1.0, ureg.mole),
        scale=Quantity(0.1, ureg.mole),
    )

    species = Species(name='water')

    substance = species * quantity_distribution

    print(substance)
