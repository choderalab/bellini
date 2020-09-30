import pytest

def test_import():
    from bellini.story import Story

def test_init():
    from bellini.story import Story
    story = Story()

def test_construct():
    from bellini import Quantity, Species, Substance, Story
    import pint
    ureg = pint.UnitRegistry()

    story = Story()

    water = Species(name='water')
    story['one_water'] = Substance(
        species=water,
        moles=Quantity(1.0, ureg.mole)
    )

    story['another_water'] = Substance(
        species=water,
        moles=Quantity(1.0, ureg.mole)
    )

    # story['combined_water'] = story.one_water + story.another_water

    story.combined_water = story.one_water + story.another_water

    assert story['combined_water'] == Substance(
        species=water,
        moles=Quantity(2.0, ureg.mole)
    )
