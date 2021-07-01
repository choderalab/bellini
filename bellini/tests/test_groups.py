import pytest
import bellini

def test_import():
    from bellini.groups import Group, Species, Substance, Mixture

def test_species():
    from bellini.groups import Species
    water = Species(name='water')

def test_species_eq():
    from bellini.groups import Species
    water0 = Species(name='water')
    water1 = Species(name='water')
    assert water0 == water1

def test_substance():
    import bellini.units as units
    from bellini.groups import Species, Substance
    from bellini.quantity import Quantity as Q
    ureg = units.ureg

    water = Species(name='water')
    one_mole = Q(1.0, unit=ureg.mole)
    one_mole_water0 = water * one_mole
    one_mole_water1 = Substance(
        species=water,
        moles=one_mole,
    )

    assert one_mole_water0 == one_mole_water1
    assert one_mole_water0 + one_mole_water1 == one_mole_water1 * Q(2.0)

def test_mixture():
    from bellini.groups import Species, Substance, Mixture
    from bellini.quantity import Quantity as Q
    import bellini.units as units
    ureg = units.ureg

    gin = Species(name='gin')
    tonic = Species(name='tonic')
    one_gin = gin * Q(1.0, unit=ureg.mole)
    one_tonic = tonic * Q(1.0, unit=ureg.mole)

    one_gin_one_tonic = one_gin + one_tonic
    assert one_gin_one_tonic * Q(2.0) == Mixture(
        [
            one_gin * Q(2.0),
            one_tonic * Q(2.0),
        ],
    )

    one_gin_two_tonic = one_gin_one_tonic + one_tonic

    assert one_gin_two_tonic == one_gin + one_tonic + one_tonic
    assert one_gin_two_tonic == one_tonic + one_tonic + one_gin
    assert one_gin_two_tonic == one_tonic + one_gin + one_tonic
