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
    import pint
    ureg = pint.UnitRegistry()

    x = Normal(
        loc=Quantity(0.0, unit='liter'),
        scale=Quantity(1.0, unit=ureg.liter),
    )

    y = Normal(
        loc=Quantity(0.0, unit='liter'),
        scale=Quantity(2.0, unit=ureg.liter),
    ).exp()

    z = x + y
