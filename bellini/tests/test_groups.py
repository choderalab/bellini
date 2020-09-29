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
    from bellini.groups import Species, Substance
    from bellini.quantity import Quantity
    import pint
    ureg = pint.UnitRegistry()
    water = Species(name='water')
    one_mole = Quantity(1.0, unit=ureg.mole)
    one_mole_water0 = water * one_mole
    one_mole_water1 = one_mole * water
    one_mole_water2 = Substance(
        species=water,
        moles=one_mole,
    )

    assert one_mole_water0 == one_mole_water1 == one_mole_water2
