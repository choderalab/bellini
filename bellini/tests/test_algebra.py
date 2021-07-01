import pytest
from bellini.groups import *
from bellini.units import *

def test_quantity_distribution_scalar():
    from bellini.distributions import Normal
    from bellini.quantity import Quantity

    x = Normal(
        loc=Quantity(1.0, ureg.liter),
        scale=Quantity(1.0, ureg.liter),
    )

    y = Quantity(
        1.0,
        ureg.liter
    )

    z = [
        x + y,
        y + x,
        x - y,
        y - x,
        x * y,
        y * x,
        x / y,
        y / x,
        x ** y,
        y ** x,
    ]

def test_quantity_distribution_arr():
    from bellini.distributions import Normal
    from bellini.quantity import Quantity
    import numpy as np

    x = Normal(
        loc=Quantity(np.zeros(3), ureg.liter),
        scale=Quantity(np.ones(3), ureg.liter),
    )

    y = Quantity(
        np.zeros(3),
        ureg.liter
    )

    z = [
        x + y,
        y + x,
        x - y,
        y - x,
        x * y,
        y * x,
        x / y,
        y / x,
        x ** y,
        y ** x,
    ]

def test_quantity_scalar_group_scalar():
    from bellini.distributions import Normal
    from bellini.quantity import Quantity as Q

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
    from bellini.distributions import Normal
    from bellini.quantity import Quantity as Q
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
    from bellini.distributions import Normal
    from bellini.quantity import Quantity as Q
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
    from bellini.distributions import Normal
    from bellini.quantity import Quantity as Q

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
    from bellini.distributions import Normal
    from bellini.quantity import Quantity as Q

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
    from bellini.distributions import Normal
    from bellini.quantity import Quantity as Q

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
