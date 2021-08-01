""" Module that contains the internal uniting system used for computations.
The internal units can be changed as necessary, although computations should be
the same regardless of what internal units system chosen.

.. attribute:: ureg

    The package-wide UnitRegistry used to specify units. Being from :code:`pint`, be
    careful not to create :code:`pint.quantity.Quantity` instead of
    :code:`bellini.quantity.Quantity`, as these do not behave the same:

    .. highlight:: python
    .. code-block:: python

        from bellini.units import ureg
        from bellini.quantity import Quantity as Q
        pint_quantity = 1 * ureg.mole # type(pint_quantity) = pint.quantity.Quantity
        bellini_quantity = Q(1, ureg.mole) # type(bellini_quantity) = bellini.quantity.Quantity

"""
# TODO: make this docstring automatic? 

# =============================================================================
# IMPORTS
# =============================================================================
import warnings
import pint
ureg = pint.UnitRegistry()

# =============================================================================
# CONSTANTS
# =============================================================================
QUANTITY_UNIT = ureg.mole
VOLUME_UNIT = ureg.liter
MASS_UNIT = ureg.gram
CONCENTRATION_UNIT = ureg.mole / ureg.liter

UNITS = [
    QUANTITY_UNIT,
    VOLUME_UNIT,
    MASS_UNIT,
    CONCENTRATION_UNIT,
]

def get_internal_units(q):
    """ Given united quantity `q`, returns the internal units corresponding to the
    units of `q` """
    dim = q.dimensionality
    for unit in UNITS:
        if dim == unit.dimensionality:
            return unit
    return q.to_base_units().units

def to_internal_units(q):
    """ Given united quantity `q`, returns `q` converted to its internal units """
    internal_unit = get_internal_units(q)
    return q.to_units(internal_unit)
