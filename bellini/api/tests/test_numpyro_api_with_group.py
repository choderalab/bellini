import pytest

def test_construct_group():
    from bellini.groups import Species, Substance, Mixture
    from bellini.quantity import Quantity
    from bellini.distributions import Normal
    from bellini.story import Story

    import pint
    ureg = pint.UnitRegistry()
    gin = Species(name='gin')
    tonic = Species(name='tonic')

    one_gin = gin * Normal(
        Quantity(1.0, unit=ureg.mole),
        Quantity(0.01, unit=ureg.mole),
        name="one_gin_volume"
    )

    one_tonic = tonic * Normal(
        Quantity(1.0, unit=ureg.mole),
        Quantity(0.05, unit=ureg.mole),
        name="one_tonic_volume"
    )

    gin_and_tonic = one_gin + one_tonic

    s = Story()

    s.gin_and_tonic = gin_and_tonic

    from bellini.api._numpyro import graph_to_numpyro_model
    model = graph_to_numpyro_model(s.g)

test_construct_group()
