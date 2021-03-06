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
        self._name = name

    @property
    def name(self):
        if self._name is not None:
            return self._name
        else:
            return repr(self)

    @name.setter
    def name(self, x):
        assert isinstance(x, str)
        self._name = x

    def _build_graph(self):
        import networkx as nx

        g = nx.MultiDiGraph() # initialize empty graph

        # loop through values
        for name, value in self.values.items():

            g.add_node(
                value,
                name=name,
            )

        # NOTE:
        # certain quantities are independent, some are dependent

        # TODO: # from JDC
        # bake that in!!!

        # TODO: # from JDC
        # support the other case, to support reactions
        # mass conservation
        # equilibrium ratio

        # three different
        # simple concentration
        # equilibirium conc, multiple conc. -> multiple conc. # with
        # observation model

        if self.laws is not None:
            for _from, _to, _lamb in self.laws:
                g.add_edge(
                    getattr(self, _from),
                    getattr(self, _to),
                    law=_lamb,
                )

        self._g = g
        return g

    @property
    def g(self):
        if not hasattr(self, "_g"):
            self._build_graph()

        if hasattr(self, "_g"):
            if self._g is None:
                self._build_graph()

        return self._g

    def __getattr__(self, name):
        if name in self.values:
            return self.values[name]

        else:
            super(Group, self).__getattribute__(name)

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

    def add_law(self, law):
        assert len(law) == 3
        _from, _to, _lamb = law
        assert isinstance(_from, str)
        assert isinstance(_to, str)
        assert isinstance(_lamb, str)
        self.laws.append(law)

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
        assert isinstance(moles, Quantity) or isinstance(moles, Distribution)

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
        assert isinstance(x, float) or isinstance(x, Distribution)

        return Substance(
            self.species,
            x * self.moles,
        )

    def __hash__(self):
        return hash(self.moles.magnitude) + hash(self.species)

class Mixture(Group):
    """ A simple mixture of substances. """
    def __init__(self, substances, **values):
        super(Mixture, self).__init__(substances=tuple(set(substances)), **values)

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
        return list(set(self.substances)) == list(set(self.substances))
