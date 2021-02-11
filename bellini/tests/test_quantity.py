import pytest

def test_one_mole():
    from bellini.quantity import Quantity
    q = Quantity(
        1.0, "mole"
    )
