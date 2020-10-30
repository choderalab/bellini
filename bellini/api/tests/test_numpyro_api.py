import pytest

def test_construct():
    import bellini
    from bellini import Quantity, Species, Substance, Story
    import pint
    ureg = pint.UnitRegistry()

    s = Story()
    water = Species(name='water')
    s.one_water_quantity = bellini.distributions.Normal(
        loc=Quantity(3.0, ureg.mole),
        scale=Quantity(0.01, ureg.mole),
    )
    s.another_water_quantity = bellini.distributions.Normal(
        loc=Quantity(3.0, ureg.mole),
        scale=Quantity(0.01, ureg.mole),
    )
    s.combined_water = s.one_water_quantity + s.another_water_quantity
    s.combined_water.observed = True

    from bellini.api._numpyro import graph_to_numpyro_model
    graph_to_numpyro_model(s.g)
