import pytest

def test_dist_dist_mix():
    from bellini.units import ureg
    import bellini.api.functional as F
    from bellini.quantity import Quantity as Q
    import numpy as np

    assert F.power(3,3) == 27

    a = Q(3, ureg.mole)
    b = Q(3)
    c = F.power(a, b)
    assert c == Q(27, ureg.mole ** 3)

    a = Q(np.arange(3), ureg.mole)
    b = Q(3)
    c = F.power(a, b)
    assert c == Q(np.arange(3) ** 3, ureg.mole ** 3)

def test_scalar_scalar_mix():
    from bellini.units import ureg
    import bellini.api.functional as F
    from bellini.quantity import Quantity as Q
    import numpy as np

    a = Q(3, ureg.mole)
    b = F.power(a, Q(2))
    assert b == Q(9, ureg.mole ** 2)

    a = Q(np.arange(3), ureg.mole)
    b = F.power(a, Q(2))
    assert b == Q(np.arange(3) ** 2, ureg.mole ** 2)

def test_scalar_dist_mix():
    from bellini.units import ureg
    import bellini.api.functional as F
    from bellini.quantity import Quantity as Q
    from bellini.distributions import Normal
    import numpy as np

    a = Normal(
        Q(np.arange(3), ureg.mole),
        Q(1, ureg.mole)
    )
    b = F.power(a, Q(2))
