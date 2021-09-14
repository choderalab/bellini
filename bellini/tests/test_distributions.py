import pytest

def test_import():
    from bellini import distributions

def test_init():
    from bellini.distributions import Normal
    from bellini import Quantity
    import pint
    ureg = pint.UnitRegistry()

    x = Normal(
        loc=Quantity(0.0, unit='liter'),
        scale=Quantity(1.0, unit=ureg.liter),
    )

def test_mix():
    from bellini.distributions import Normal
    from bellini import Quantity
    import bellini.api.functional as F
    from bellini.units import ureg
    import bellini
    bellini.verbose = True

    x = Normal(
        loc=Quantity(0.0),
        scale=Quantity(1.0),
    )

    y = F.exp(Normal(
        loc=Quantity(0.0),
        scale=Quantity(2.0),
    ))

    z = x + y
