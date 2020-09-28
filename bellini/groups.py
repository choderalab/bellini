# =============================================================================
# IMPORTS
# =============================================================================
import abc
import pint
import numpy as np
import torch
import bellini

# =============================================================================
# BASE CLASS
# =============================================================================
class Group(abc.ABC):
    """ Base class for groups that hold quantities and children. """
    allowed_instances = (bellini.quantity.Quantity)

    def __init__(self, name, laws=None, **quantities):
        self.name = name

        # sanity check quantities
        assert all(
            isinstance(quantity, self.allowed_instances)
            for quantity in quantities), "quantity instance is not allowed."

        self.quantities = quantities
        self.laws = laws

    def __getattr__(self, name):
        if name in self.quantities:
            return self.quantities[name]

        else:
            AttributeError(
                "%s has no attribute %s" % (
                    self.name,
                    name,
                ))

    def apply_laws(self):
        if self.laws is not None:
            for law in self.laws:
                new_quantities = law(self)
                for name, quantity in new_quantities:
                    setattr(self, name, quantity)

# =============================================================================
# SUBCLASSES
# =============================================================================
class Species(Group):
    """ A chemical species without mass. """
    def __init__(self, name, **quantities):
        super(Species, self).__init__(name=name, **quantities)

class Substance(Group):
    """ A chemical substance with species and quantities. """
    def __init__(self, name, species, moles, **quantities):
        # check the type of species
        assert isinstance(
            species,
            bellini.species.Species
        )

        assert isinstance(
            moles,
            bellini.quantity.Quantity,
        )

        super(Substance, self).__init__(
            name=name,
            species=species,
            moles=moles,
            **quantities
        )

class Mixture(Group):
    """ A simple mixture of substances. """
    allowed_instances = (bellini.quantity.Quantity, list)

    def __init__(self, name, substances):
        super(Mixture, self).__init__(name=name, substances=substances)
