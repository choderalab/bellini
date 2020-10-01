import pytest

def test_init():
    from bellini import Substance, Quantity, Species
    from bellini.distributions import Normal
    import pint
    ureg = pint.UnitRegistry()

    quantity_distribution = Normal(
        loc=Quantity(1.0, ureg.mole),
        scale=Quantity(0.1, ureg.mole ** 0.5),
    )

    species = Species(name='water')

    substance = Species * quantity_distribution

    print(substance)
