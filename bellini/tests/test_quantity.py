import pytest

def test_new_quantity():
    import bellini
    import pint
    ureg = pint.UnitRegistry()

    volume = bellini.quantity.Quantity(
        1.0,
        ureg.liter
    )

def test_change_mutable():
    import bellini
    import pint
    ureg = pint.UnitRegistry()

    volume = bellini.quantity.Quantity(
        1.0,
        ureg.liter
    )

    volume.mutable = True

def test_new_quantity_torch():
    import bellini
    import pint
    import torch

    ureg = pint.UnitRegistry()

    volume = bellini.quantity.Quantity(
        torch.ones(4, 3),
        ureg.liter
    )
