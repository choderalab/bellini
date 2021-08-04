""" Groups are objects that represent experimental reagents, e.g. compounds,
buffers, solvents, etc.

The key thing to remember is that Groups are not immutable e.g. any operation
using groups will produce new Groups rather than modifying existing Groups'
parameters.
"""


# =============================================================================
# IMPORTS
# =============================================================================
import abc
import pint
import numpy as np
import torch
from bellini.laws import Law
from bellini.quantity import Quantity
from bellini.distributions import Distribution
from bellini.api import utils
from bellini.units import ureg, VOLUME_UNIT, CONCENTRATION_UNIT
from collections import defaultdict

# =============================================================================
# BASE CLASS
# =============================================================================

class Group(abc.ABC):
    """ Base class for groups that hold quantities and children. """

    def __init__(self, name=None, **values):
        self.values = values
        self._name = name

    @abc.abstractmethod
    def copy(self):
        """ Return a copy of itself """
        raise NotImplementedError()

    @property
    def name(self):
        """ A string used to represent the Group """
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

        # to compose or to not to compose?
        for name, value in self.values.items():
            if isinstance(value, (list, tuple)):
                for v in value:
                    #print(v)
                    g.add_node(
                        v,
                        name=name
                    )
                    g = nx.compose(g, v.g)
            elif isinstance(value, dict):
                for k, v in value.items():
                    #print(v)
                    g.add_node(
                        v,
                        name=f"{name}[{k}]"
                    )
                    g = nx.compose(g, v.g)
            else:
                #print(value)
                g.add_node(
                    value,
                    name=name
                )
                g = nx.compose(g, value.g)

        self._g = g
        return g

    @property
    def g(self):
        """ A networkx graph representing how the group was constructed """
        if not hasattr(self, "_g"):
            self._build_graph()

        if hasattr(self, "_g"):
            if self._g is None:
                self._build_graph()

        return self._g

    def __getattr__(self, name):
        if name in self.values.keys():
            return self.values[name]
        else:
            return super().__getattribute__(name)

    def _register(self, name, item):
        self.values[name] = item

    def __eq__(self, new_group):
        return {
                **self.values,
                'name': self.name
            } ==  {
                **new_group.values,
                'name': new_group.name
            }

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

class LawedGroup(Group):
    """ Class constructed by default for a Group after a Law has been applied to it """
    def __new__(cls, group, law):
        assert isinstance(group, Group)
        assert isinstance(law, Law)
        return group.copy()

    def __init__(self, group, law):
        super().__init__(
            base_group = group,
            law = law
        )

    def _build_graph(self):
        import networkx as nx # local import
        g = nx.MultiDiGraph() # new graph
        g.add_node(
            self,
            ntype="lawed_group",
            law=law,
        )
        g.add_node(
            self.base_group,
            ntype="base_group",
        )
        g.add_edge(
            self.base_group,
            self,
            etype="is_base_group_of"
        )
        g = nx.compose(g, self.base_group.g)
        self._g = g
        return g

# =============================================================================
# Chemicals
# =============================================================================

class Chemical(Group):
    """ Base class for all chemical-like Groups """
    @abc.abstractmethod
    def __add__(self, x):
        raise NotImplementedError

    def __radd__(self, x):
        return self.__add__(x)

    @abc.abstractmethod
    def __mul__(self, x):
        raise NotImplementedError

    def __rmul__(self, x):
        return self.__mul__(x)

    def _sanitize(self, x):
        if utils.is_scalar(x):
            x = Quantity(x, ureg.dimensionless)
        assert isinstance(x, (Quantity, Distribution))
        assert x.units.dimensionality == ureg.dimensionless.dimensionality
        return x

class Species(Chemical):
    """ A chemical species without mass. """
    def __init__(self, name, **values):
        super().__init__(name=name, **values)

    def copy(self):
        return Species(
            name=self.name,
            **self.values
        )

    def __add__(self, x):
        return NotImplemented

    def __mul__(self, quantity):
        """ Species times quantity equals substance. """
        # check quantity is in mole
        assert isinstance(quantity, (Quantity, Distribution))
        if quantity.units.dimensionality == ureg.mole.dimensionality:
            return Substance(
                species=self,
                moles=quantity,
            )
        elif quantity.units.dimensionality == VOLUME_UNIT.dimensionality:
            return Solvent(
                species=self,
                volume=quantity
            )
        else:
            raise ValueError(f"{self} and {quantity} cannot be multiplied")

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)


class Substance(Chemical):
    """ A chemical substance with species and number of moles. """
    def __init__(self, species, moles, **values):
        # check the type of species
        assert isinstance(
            species,
            Species,
        )

        super().__init__(
            species=species,
            moles=moles,
            **values
        )

    def copy(self):
        return Substance(
            **self.values
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
        else:
            raise ValueError("Can only add Mixture or Substance to Substance")

    def __mul__(self, x):
        x = self._sanitize(x)

        return Substance(
            self.species,
            x * self.moles,
        )

    def __hash__(self):
        return hash(self.moles) + hash(self.species)

    def __getitem__(self, idxs):
        assert utils.is_arr(self.moles)
        return Substance(
            self.species,
            self.moles[idxs]
        )


class Liquid(Chemical):
    """ Base class for all liquid-like Chemicals """
    @abc.abstractmethod
    def aliquot(self, volume):
        """ Return an aliquot of itself, as well as the remaining source solution

        Parameters
        ----------
        volume : Quantity (volume)
            The amount of liquid to aliquot out of the current solution

        Returns
        -------
        aliquot: Liquid
            The aliquot drawn from the initial solution
        source: Liquid
            The remaining solution after the aliquot has been removed
        """
        raise NotImplementedError()

class Solvent(Liquid):
    """ A chemical substance with species and volume. """
    def __init__(self, species, volume, **values):
        # check the type of species
        assert isinstance(
            species,
            Species,
        )

        super().__init__(
            species=species,
            volume=volume,
            **values
        )

    def copy(self):
        return Solvent(
            **self.values
        )

    def __repr__(self):
        return '%s of %s' % (str(self.volume), str(self.species))

    def __add__(self, x):

        if isinstance(x, Solvent):
            if x.species == self.species:
                return Solvent(
                    self.species,
                    self.volume + x.volume
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
        else:
            raise ValueError("Can only add Solvent or Mixture to Ssolvent")

    def __mul__(self, x):
        x = self._sanitize(x)

        return Solvent(
            self.species,
            x * self.volume,
        )

    def __hash__(self):
        return hash(self.volume) + hash(self.species)

    def aliquot(self, volume):
        """ Return an aliquot of itself, as well as the remaining source solution

        Parameters
        ----------
        volume : Quantity (volume)
            The amount of liquid to aliquot out of the current solution

        Returns
        -------
        aliquot: Solvent
            The aliquot drawn from the initial solution
        source: Solvent
            The remaining solution after the aliquot has been removed
        """
        #assert volume.units == VOLUME_UNIT

        aliquot = Solvent(
            species=self.species,
            volume=volume
        )

        source = Solvent(
            species=self.species,
            volume=self.volume - volume
        )

        return aliquot, source

    def __getitem__(self, idxs):
        assert utils.is_arr(self.volume)
        return Solvent(
            self.species,
            self.volume[idxs]
        )


class Mixture(Chemical):
    """ A simple mixture of substances. """
    def __init__(self, substances, **values):

        sub_dict = {}
        shape = None
        for sub in substances:
            if utils.is_arr(sub.moles) and shape is None:
                shape = sub.moles.shape
            elif shape and utils.is_arr(sub.moles):
                assert sub.moles.shape == shape, "shape of all substance Quantities must be the same"
            elif shape and not utils.is_arr(sub.moles):
                raise ValueError("if mixture contains array Substance, all substance Quantities must be arrays")

            species = sub.species
            if species not in sub_dict.keys():
                sub_dict[species] = sub
            else:
                sub_dict[species] = sub_dict[species] + sub

        super().__init__(substances=tuple(sub_dict.values()), **values)

    def copy(self):
        return Mixture(
            **self.values
        )

    def __repr__(self):
        return ' and '.join([str(x) for x in self.substances])

    def __mul__(self, x):
        x = self._sanitize(x)

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

        else:
            raise ValueError(f"{self} and {x} cannot be added")

    def __eq__(self, x):
        return list(set(self.substances)) == list(set(self.substances))

    def __hash__(self):
        return sum([
            hash(sub.moles)
            + hash(sub.species)
            for sub in self.substances
        ])

    def __getitem__(self, idxs):
        substances = [
            Substance(
                species=substance.species,
                moles=substance.moles[idxs],
            ) for substance in self.substances
        ]
        return Mixture(substances=substances)

class Solution(Liquid):
    """ A substance or a mixture dissolved in a solvent """
    def __init__(self, mixture, solvent, **values):
        # check the type of substance and solvent
        if isinstance(mixture, Substance):
            mixture = Mixture([mixture])

        assert isinstance(
            mixture,
            Mixture,
        )
        assert isinstance(
            solvent,
            Solvent
        )

        super().__init__(
            mixture=mixture,
            solvent=solvent,
            **values
        )

        # compute _concentrations
        # since otherwise two solutions that are the same might
        # register as different
        _ = self.concentrations

    @classmethod
    def _empty_dict_attr(self):
        def zero_conc():
            return Quantity(0, CONCENTRATION_UNIT)
        return defaultdict(zero_conc)

    def add_dict_attr(self, name):
        self._register(name, Solution._empty_dict_attr())

    def copy(self):
        return Solution(
            **self.values
        )

    @property
    def moles(self):
        """ The number of moles of each substance in the solution """
        return [substance.moles for substance in self.mixture.substances]

    @property
    def volume(self):
        """ The volume of the solution """
        return self.solvent.volume

    @property
    def concentration(self):
        """ The concentration of the solution, if there is only one solute dissolved """
        assert len(self.concentrations) == 1, f"{self} complex solution, use `self.concentrations` instead"
        return list(self.concentrations.values())[0]

    @property
    def concentrations(self):
        """ The concentrations of all solutes dissolved """
        if not hasattr(self, "_concentrations"):
            self.add_dict_attr("_concentrations")
            self._concentrations.update({
                substance.species: substance.moles/self.solvent.volume
                for substance in self.mixture.substances
            })
        return self._concentrations

    def __repr__(self):
        return " and ".join([
        f"{self.concentrations[substance.species]} of {substance.species}"
        for substance in self.mixture.substances
        ]) + f" in {self.solvent}"

    def __add__(self, x):
        if isinstance(x, Solvent):
            assert self.solvent.species == x.species, "currently don't suppose mixed solvent solutions"
            return Solution(
                mixture = self.mixture,
                solvent = self.solvent + x
            )
        elif isinstance(x, Solution):
            assert self.solvent.species == x.solvent.species, "currently don't suppose mixed solvent solutions"
            return Solution(
                mixture = self.mixture + x.mixture,
                solvent = self.solvent + x.solvent
            )
        else:
            raise NotImplementedError(f"adding between {type(self)} and {type(x)} not supported")

    def __mul__(self, x):
        x = self._sanitize(x)

        return Solution(
            mixture = x * self.mixture,
            solvent = x * self.solvent
        )

    def __rmul__(self, x):
        return self.__mul__(x)

    def __getitem__(self, idxs):
        return Solution(
            mixture=self.mixture[idxs],
            solvent=self.solvent[idxs]
        )

    def aliquot(self, volume):
        """ Return an aliquot of itself, as well as the remaining source solution

        Parameters
        ----------
        volume : Quantity (volume)
            The amount of liquid to aliquot out of the current solution

        Returns
        -------
        aliquot: Solution
            The aliquot drawn from the initial solution
        source: Solution
            The remaining solution after the aliquot has been removed
        """
        #assert volume.units == VOLUME_UNIT

        new_volume = self.volume - volume

        aliquot_mixture = Mixture(
            substances = [
                Substance(
                    species = substance.species,
                    moles = self.concentrations[substance.species] * volume
                ) for substance in self.mixture.substances
            ]
        )

        aliquot_solvent, source_solvent = self.solvent.aliquot(volume)

        aliquot = Solution(
            mixture = aliquot_mixture,
            solvent = aliquot_solvent,
            concentrations = self.concentrations
        )

        source_mixture = Mixture(
            substances = [
                Substance(
                    species = substance.species,
                    moles = self.concentrations[substance.species] * new_volume
                ) for substance in self.mixture.substances
            ]
        )

        source = Solution(
            mixture = source_mixture,
            solvent = source_solvent,
            concentrations = self.concentrations
        )

        return aliquot, source
