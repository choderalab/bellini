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

def test_pipeline():
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

    import networkx as nx
    edges = list(nx.bfs_edges(s.g, source=s.combined_water, reverse=True))[::-1]
    print(edges)
