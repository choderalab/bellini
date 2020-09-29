# =============================================================================
# IMPORTS
# =============================================================================
import abc
import pint
import numpy as np
import torch
import bellini
from bellini.quantity import Quantity
from bellini.distributions import Distribution
ureg = pint.UnitRegistry()

# =============================================================================
# BASE CLASS
# =============================================================================
class Group(abc.ABC):
    """ Base class for groups that hold quantities and children. """
    allowed_instances = (
        Quantity, Distribution,
    )

    def __init__(self, name=None, laws=None, **values):

        # sanity check quantities
        assert all(
            isinstance(value, self.allowed_instances)
            for value in values), "input instance is not allowed."

        self.values = values
        self.laws = laws
        self.name = name

    def __getattr__(self, name):
        if name in self.values:
            return self.values[name]

        else:
            AttributeError(
                "%s has no attribute %s" % (
                    self.__class__.__name__,
                    name,
                ))

    def __eq__(self, new_group):
        return {
                **self.values,
                'name': self.name, 'laws': self.laws
            } ==  {
                **new_group.values,
                'name': new_group.name, 'laws': new_group.laws
            }

    def apply_laws(self):
        if self.laws is not None:
            for law in self.laws:
                new_values = law(self)
                for name, value in new_values:
                    setattr(self, name, value)

# =============================================================================
# SUBCLASSES
# =============================================================================
class Species(Group):
    """ A chemical species without mass. """
    def __init__(self, name, **values):
        super(Species, self).__init__(name=name, **values)

    def __mul__(self, moles):
        """ Species times quantity equals substance. """
        # check quantity is in mole
        assert isinstance(
            moles,
            Quantity
        )

        assert moles.unit.is_compatible_with(
            ureg.mole
        )

        return Substance(
            species=self,
            moles=moles,
        )

class Substance(Group):
    """ A chemical substance with species and quantities. """
    def __init__(self, species, moles, **values):
        # check the type of species
        assert isinstance(
            species,
            Species,
        )

        assert isinstance(
            moles,
            Quantity,
        )

        super(Substance, self).__init__(
            species=species,
            moles=moles,
            **values
        )

    def __add__(self, substance):
        assert isinstance(
            substance,
            Substance,
        )

        if substance.species == self.species:
            return Substance(
                self.species,
                self.moles + substance.moles
            )

        else:
            return Mixture(
                [
                    self,
                    substance,
                ]
            )

class Mixture(Group):
    """ A simple mixture of substances. """
    allowed_instances = (list, )

    def __init__(self, substances, **values):
        # sanity check quantities
        assert all(
            isinstance(substance, Substance)
            for substance in substances), "input instance is not allowed."

        super(Mixture, self).__init__(substances=substances, **values)
