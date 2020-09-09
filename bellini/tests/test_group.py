import pytest

def test_new_group():
    import bellini as be
    group = be.group.Group()


def test_group_assign():
    import bellini as be
    import pint
    ureg = pint.UnitRegistry()

    group = be.group.Group()

    group['volume'] = be.quantity.Quantity(
        1.0,
        ureg.liter
    )

def test_group_assign_group():
    import bellini as be
    import pint
    ureg = pint.UnitRegistry()

    group = be.group.Group()

    group['volume'] = be.quantity.Quantity(
        1.0,
        ureg.liter
    )

    group['self'] = group

def test_raise_error():
    with pytest.raises(AssertionError):
        import bellini as be
        import pint
        ureg = pint.UnitRegistry()

        group = be.group.Group()

        group['wrong'] = 1.0
