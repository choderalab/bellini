import pytest

def test_new_quantity():
    import bellini as be
    import pint
    ureg = pint.UnitRegistry()

    volume = be.quantity.Quantity(
        1.0,
        ureg.liter
    )

def test_change_mutable():
    import bellini as be
    import pint
    ureg = pint.UnitRegistry()

    volume = be.quantity.Quantity(
        1.0,
        ureg.liter
    )

    volume.mutable = True    

def test_new_quantity_torch():
    import bellini as be
    import pint
    import torch

    ureg = pint.UnitRegistry()

    volume = be.quantity.Quantity(
        torch.ones(4, 3),
        ureg.liter
    )
