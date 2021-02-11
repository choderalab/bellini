import pytest

def test_delta():
    from bellini.quantity import Quantity
    from bellini.distributions import Delta
    delta = Delta(Quantity(1.0, "mole"))
    print(delta)

def test_normal():
    from bellini.distributions import Normal
    from bellini.quantity import Quantity
    normal = Normal(Quantity(1.0, "mole"), Quantity(1.0, "mole"))
    print(normal)
