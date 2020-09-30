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

    def __init__(self, name=None, laws=None, **values):
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

        assert moles.units.dimensionality == ureg.mole.dimensionality

        return Substance(
            species=self,
            moles=moles,
        )

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

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

    def __repr__(self):
        return '%s of %s' % (str(self.moles), str(self.species))

    def __add__(self, x):

        if isinstance(x, Substance):
            if x.species == self.species:
                return Substance(
                    self.species,
                    self.moles + x.moles
                )

            else:
                return Mixture(
                    [
                        self,
                        x,
                    ]
                )

        elif isinstance(x, Mixture):
            return x + self

    def __mul__(self, x):
        assert isinstance(x, float)

        return Substance(
            self.species,
            x * self.moles,
        )

    def __hash__(self):
        return hash(self.moles.magnitude) + hash(self.species)

class Mixture(Group):
    """ A simple mixture of substances. """
    def __init__(self, substances, **values):
        super(Mixture, self).__init__(substances=set(substances), **values)

    def __repr__(self):
        return ' and '.join([str(x) for x in self.substances])

    def __mul__(self, x):
        assert isinstance(x, float)

        return Mixture(
            [
                Substance(
                    species=substance.species,
                    moles=x * substance.moles,
                ) for substance in self.substances
            ]
        )

    def __add__(self, x):

        if isinstance(x, Substance):
            substances = [
                Substance(
                    species=substance.species,
                    moles=substance.moles,
                ) for substance in self.substances
            ]

            # print([substance.species for substance in substances])

            new_substance = True
            for idx, substance in enumerate(substances):
                if x.species == substance.species:
                    substances[idx] = Substance(
                        substance.species,
                        substance.moles + x.moles
                    )
                    new_substance = False

            if new_substance is True:
                substances.append(x)

            return Mixture(substances=substances)

        elif isinstance(x, Mixture):
            mixture = self
            for substance in x.substances:
                mixture = mixture + substance
            return mixture

    def __eq__(self, x):
        return self.substances == self.substances
