import pytest

def test_quantity_dist_scalar_unitless():
    from bellini.units import ureg
    from bellini.distributions import Normal, ComposedDistribution, TransformedDistribution
    from bellini.quantity import Quantity
    import bellini.api.functional as F

    x = Normal(
        loc=Quantity(1.0, ureg.dimensionless),
        scale=Quantity(1.0, ureg.dimensionless),
    )

    y = Quantity(
        1.0,
        ureg.dimensionless
    )

    composed_dists = [
        x + y,
        y + x,
        x - y,
        y - x,
        x * y,
        y * x,
        x / y,
        y / x,
    ]

    for res in composed_dists:
        assert isinstance(res, ComposedDistribution)
        assert res.dimensionality == ureg.dimensionless.dimensionality

    transformed_dists = [
        x ** y,
        y ** x,
        F.exp(x),
        F.log(x)
    ]

    for res in transformed_dists:
        assert isinstance(res, TransformedDistribution)
        assert res.dimensionality == ureg.dimensionless.dimensionality


def test_quantity_dist_scalar_united():
    from bellini.units import ureg
    from bellini.distributions import Normal, ComposedDistribution, TransformedDistribution
    from bellini.quantity import Quantity
    import bellini.api.functional as F
    from pint.errors import DimensionalityError

    x = Normal(
        loc=Quantity(1.0, ureg.mole),
        scale=Quantity(1.0, ureg.mole),
    )

    y = Quantity(
        1.0,
        ureg.mole
    )

    composed_dists_mole = [
        x + y,
        y + x,
        x - y,
        y - x,
    ]

    for res in composed_dists_mole:
        assert isinstance(res, ComposedDistribution)
        assert res.dimensionality == ureg.mole.dimensionality

    composed_dists_mole_squared = [
        x * y,
        y * x,
    ]

    for res in composed_dists_mole_squared:
        assert isinstance(res, ComposedDistribution)
        assert res.dimensionality == (ureg.mole**2).dimensionality

    composed_dists_unitless = [
        x / y,
        y / x,
    ]

    for res in composed_dists_unitless:
        assert isinstance(res, ComposedDistribution)
        assert res.dimensionality == ureg.dimensionless.dimensionality

    def op_fails(fn):
        try:
            res = fn(x, y)
            return False
        except DimensionalityError as e:
            return True

    transformed_dists_fns = [
        lambda x,y: x ** y,
        lambda x,y: y ** x,
        lambda x,y: F.exp(x),
        lambda x,y: F.log(x)
    ]

    for fn in transformed_dists_fns:
        assert op_fails(fn)


def test_quantity_dist_arr_unitless():
    from bellini.distributions import Normal, ComposedDistribution, TransformedDistribution
    from bellini.quantity import Quantity
    from bellini.units import ureg
    import bellini.api.functional as F
    import numpy as np

    x = Normal(
        loc=Quantity(np.zeros(3), ureg.dimensionless),
        scale=Quantity(np.ones(3), ureg.dimensionless),
    )

    y = Quantity(
        np.zeros(3),
        ureg.dimensionless
    )

    composed_dists = [
        x + y,
        y + x,
        x - y,
        y - x,
        x * y,
        y * x,
        x / y,
        y / x,
    ]

    for res in composed_dists:
        assert isinstance(res, ComposedDistribution)
        assert res.dimensionality == ureg.dimensionless.dimensionality

    transformed_dists = [
        x ** y,
        y ** x,
        F.exp(x),
        F.log(x)
    ]

    for res in transformed_dists:
        assert isinstance(res, TransformedDistribution)
        assert res.dimensionality == ureg.dimensionless.dimensionality

def test_quantity_scalar_group_scalar():
    from bellini.groups import Species, Solution
    from bellini.distributions import Normal
    from bellini.quantity import Quantity as Q
    from bellini.units import ureg

    sugar = Species("sugar")
    salt = Species("salt")
    water = Species("water")
    one_mol_sugar = sugar * Q(1, ureg.mole)
    one_mol_sugar = Q(1, ureg.mole) * sugar
    one_mol_salt = salt * Q(1, ureg.mole)
    one_mol_sugar = Q(1, ureg.mole) * salt
    one_liter_water = water * Q(1, ureg.liter)
    sugar_water = Solution(one_mol_sugar, one_liter_water)
    sugar_n_salt = one_mol_sugar + one_mol_salt
    salty_sugar_water = Solution(sugar_n_salt, one_liter_water)

    y = Q(1)
    z = [
        one_mol_sugar * y,
        y * one_mol_sugar,
        y * sugar_water,
        sugar_water * y,
        sugar_n_salt * y,
        y * sugar_n_salt,
        y * salty_sugar_water,
        salty_sugar_water * y
    ]

def test_quantity_scalar_group_arr():
    from bellini.groups import Species, Solution
    from bellini.distributions import Normal
    from bellini.quantity import Quantity as Q
    from bellini.units import ureg
    import numpy as np

    sugar = Species("sugar")
    salt = Species("salt")
    water = Species("water")
    one_mol_sugar = sugar * Q(np.ones(3), ureg.mole)
    one_mol_sugar = Q(np.ones(3), ureg.mole) * sugar
    one_mol_salt = salt * Q(np.ones(3), ureg.mole)
    one_mol_sugar = Q(np.ones(3), ureg.mole) * salt
    one_liter_water = water * Q(np.ones(3), ureg.liter)
    sugar_water = Solution(one_mol_sugar, one_liter_water)
    sugar_n_salt = one_mol_sugar + one_mol_salt
    salty_sugar_water = Solution(sugar_n_salt, one_liter_water)

    y = Q(1)
    z = [
        one_mol_sugar * y,
        y * one_mol_sugar,
        y * sugar_water,
        sugar_water * y,
        sugar_n_salt * y,
        y * sugar_n_salt,
        y * salty_sugar_water,
        salty_sugar_water * y
    ]

def test_quantity_arr_group_arr():
    from bellini.groups import Species, Solution
    from bellini.distributions import Normal
    from bellini.quantity import Quantity as Q
    from bellini.units import ureg
    import numpy as np

    sugar = Species("sugar")
    salt = Species("salt")
    water = Species("water")
    one_mol_sugar = sugar * Q(np.ones(3), ureg.mole)
    one_mol_sugar = Q(np.ones(3), ureg.mole) * sugar
    one_mol_salt = salt * Q(np.ones(3), ureg.mole)
    one_mol_sugar = Q(np.ones(3), ureg.mole) * salt
    one_liter_water = water * Q(np.ones(3), ureg.liter)
    sugar_water = Solution(one_mol_sugar, one_liter_water)
    sugar_n_salt = one_mol_sugar + one_mol_salt
    salty_sugar_water = Solution(sugar_n_salt, one_liter_water)

    y = Q(np.ones(3))
    z = [
        one_mol_sugar * y,
        y * one_mol_sugar,
        y * sugar_water,
        sugar_water * y,
        sugar_n_salt * y,
        y * sugar_n_salt,
        y * salty_sugar_water,
        salty_sugar_water * y
    ]

def test_quantity_scalar_group_dist_scalar():
    from bellini.groups import Species, Solution
    from bellini.distributions import Normal
    from bellini.quantity import Quantity as Q
    from bellini.units import ureg

    x = Normal(
        loc=Q(1.0, ureg.mole),
        scale=Q(1.0, ureg.mole),
    )

    sugar = Species("sugar")
    salt = Species("salt")
    water = Species("water")
    one_mol_sugar = sugar * x
    one_mol_sugar = x * sugar
    one_mol_salt = salt * x
    one_mol_sugar = x * salt
    one_liter_water = water * Q(1, ureg.liter)
    sugar_water = Solution(one_mol_sugar, one_liter_water)
    sugar_n_salt = one_mol_sugar + one_mol_salt
    salty_sugar_water = Solution(sugar_n_salt, one_liter_water)

    y = Q(1)
    z = [
        one_mol_sugar * y,
        y * one_mol_sugar,
        y * sugar_water,
        sugar_water * y,
        sugar_n_salt * y,
        y * sugar_n_salt,
        y * salty_sugar_water,
        salty_sugar_water * y
    ]

def test_quantity_scalar_group_dist_arr():
    from bellini.groups import Species, Solution
    from bellini.distributions import Normal
    from bellini.quantity import Quantity as Q
    from bellini.units import ureg
    import numpy as np

    x = Normal(
        loc=Q(np.ones(3), ureg.mole),
        scale=Q(np.ones(3), ureg.mole),
    )

    sugar = Species("sugar")
    salt = Species("salt")
    water = Species("water")
    one_mol_sugar = sugar * x
    one_mol_sugar = x * sugar
    one_mol_salt = salt * x
    one_mol_sugar = x * salt
    one_liter_water = water * Q(np.ones(3), ureg.liter)
    sugar_water = Solution(one_mol_sugar, one_liter_water)
    sugar_n_salt = one_mol_sugar + one_mol_salt
    salty_sugar_water = Solution(sugar_n_salt, one_liter_water)

    y = Q(1)
    z = [
        one_mol_sugar * y,
        y * one_mol_sugar,
        y * sugar_water,
        sugar_water * y,
        sugar_n_salt * y,
        y * sugar_n_salt,
        y * salty_sugar_water,
        salty_sugar_water * y
    ]

def test_quantity_arr_group_dist_arr():
    from bellini.groups import Species, Solution
    from bellini.distributions import Normal
    from bellini.quantity import Quantity as Q
    from bellini.units import ureg
    import numpy as np

    x = Normal(
        loc=Q(np.ones(3), ureg.mole),
        scale=Q(np.ones(3), ureg.mole),
    )

    sugar = Species("sugar")
    salt = Species("salt")
    water = Species("water")
    one_mol_sugar = sugar * x
    one_mol_sugar = x * sugar
    one_mol_salt = salt * x
    one_mol_sugar = x * salt
    one_liter_water = water * Q(np.ones(3), ureg.liter)
    sugar_water = Solution(one_mol_sugar, one_liter_water)
    sugar_n_salt = one_mol_sugar + one_mol_salt
    salty_sugar_water = Solution(sugar_n_salt, one_liter_water)

    y = Q(np.ones(3))
    z = [
        one_mol_sugar * y,
        y * one_mol_sugar,
        y * sugar_water,
        sugar_water * y,
        sugar_n_salt * y,
        y * sugar_n_salt,
        y * salty_sugar_water,
        salty_sugar_water * y
    ]
