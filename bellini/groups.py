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
from bellini.units import *
#ureg = pint.UnitRegistry()

# =============================================================================
# BASE CLASS
# =============================================================================

class Group(abc.ABC):
    """ Base class for groups that hold quantities and children. """

    def __init__(self, name=None, laws=[], **values):
        self.values = values
        self.laws = laws
        self.io_maps = []
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

        """
        # loop through values
        for name, value in self.values.items():

            g.add_node(
                value,
                name=name,
            )
        """
        for name, value in self.values.items():
            if isinstance(value, list) or isinstance(value, tuple):
                for v in value:
                    g.add_node(
                        v,
                        name=name
                    )
            elif isinstance(value, dict):
                for k, v in value.items():
                    g.add_node(
                        v,
                        name=f"{name}[{k}]"
                    )
            else:
                g.add_node(
                    value,
                    name=name
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


        for io_map in self.io_maps:
            for _from, _to in io_map:
                g.add_edge(
                    getattr(self, _from),
                    getattr(self, _to),
                    law="i need to fill this out",
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
                new_values, io_map = law.apply(self)
                self.io_maps.append(io_map)
                for name, value in new_values.items():
                    setattr(self, name, value)

    def add_law(self, law):
        assert isinstance(law, bellini.laws.Law)
        """
        _from, _to, _lamb = law
        assert isinstance(_from, dict)
        assert isinstance(_to, dict)
        assert isinstance(_lamb, str)
        """
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
    """ A chemical substance with species and number of moles. """
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
        else:
            raise ValueError("Can only add Mixture or Substance to Substance")

    def __mul__(self, x):
        assert isinstance(x, float) or isinstance(x, Distribution)

        return Substance(
            self.species,
            x * self.moles,
        )

    def __hash__(self):
        return hash(self.moles.magnitude) + hash(self.species)


class Liquid(Group):
    @abc.abstractmethod
    def aliquot(self, volume):
        raise NotImplementedError()

class Solvent(Liquid):
    """ A chemical substance with species and volume. """
    def __init__(self, species, volume, **values):
        # check the type of species
        assert isinstance(
            species,
            Species,
        )

        super(Solvent, self).__init__(
            species=species,
            volume=volume,
            **values
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
        assert isinstance(x, float) or isinstance(x, Distribution)

        return Solvent(
            self.species,
            x * self.volume,
        )

    def __hash__(self):
        return hash(self.volume.magnitude) + hash(self.species)

    def aliquot(self, volume):
        """ Split into aliquot and source """
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


class Mixture(Group):
    """ A simple mixture of substances. """
    def __init__(self, substances, **values):

        sub_dict = {}
        for sub in substances:
            species = sub.species
            if species not in sub_dict.keys():
                sub_dict[species] = sub
            else:
                sub_dict[species] = sub_dict[species] + sub

        super(Mixture, self).__init__(substances=tuple(sub_dict.values()), **values)

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

    def __hash__(self):
        return sum([
            hash(sub.moles)
            + hash(sub.species)
            for sub in self.substances
        ])


class Solution(Liquid):
    def __init__(self, substance, solvent, **values):
        # check the type of substance and solvent
        assert isinstance(
            substance,
            Substance,
        )
        assert isinstance(
            solvent,
            Solvent
        )

        if 'concentration' not in values.keys():
            values['concentration'] = substance.moles/solvent.volume

        super(Solution, self).__init__(
            substance=substance,
            solvent=solvent,
            **values
        )

    @property
    def moles(self):
        return self.substance.moles

    @property
    def volume(self):
        return self.solvent.volume

    def __repr__(self):
        return f"{self.concentration} of {self.substance.species} in {self.solvent}"

    def __add__(self, x):
        if isinstance(x, Solvent):
            assert self.solvent.species == x.species, "currently don't suppose mixed solvent solutions"
            return Solution(
                substance = self.substance,
                solvent = self.solvent + x
            )
        elif isinstance(x, Solution):
            assert self.solvent.species == x.solvent.species, "currently don't suppose mixed solvent solutions"
            if self.substance.species == x.substance.species:
                return Solution(
                    substance = self.substance + x.substance,
                    solvent = self.solvent + x.solvent
                )
            else:
                return ComplexSolution(
                    mixture = self.substance + x.substance,
                    solvent = self.solvent + x.solvent
                )
        else:
            raise NotImplementedError(f"adding between {type(self)} and {type(x)} not supported")

    def __mul__(self, x):
        assert isinstance(x, float)

        return Solution(
            substance = x * self.substance,
            solvent = x * self.solvent,
            concentration = self.concentration
        )

    def aliquot(self, volume):
        """ Split into aliquot and source """
        #assert volume.units == VOLUME_UNIT

        new_volume = self.volume - volume

        aliquot_substance = Substance(
            species=self.substance.species,
            moles=self.concentration * volume
        )

        aliquot_solvent, source_solvent = self.solvent.aliquot(volume)

        aliquot = Solution(
            substance = aliquot_substance,
            solvent = aliquot_solvent,
            concentration = self.concentration
        )

        source_substance = Substance(
            species=self.substance.species,
            moles=self.concentration * new_volume
        )

        source = Solution(
            substance = source_substance,
            solvent = source_solvent,
            concentration = self.concentration
        )

        return aliquot, source

class ComplexSolution(Liquid):
    def __init__(self, mixture, solvent, **values):
        # check the type of substance and solvent
        assert isinstance(
            mixture,
            Mixture,
        )
        assert isinstance(
            solvent,
            Solvent
        )

        if 'concentrations' not in values.keys():
            values['concentrations'] = {
                substance.species: substance.moles/solvent.volume
                for substance in mixture.substances
            }

        super(ComplexSolution, self).__init__(
            mixture=mixture,
            solvent=solvent,
            **values
        )

    @property
    def moles(self):
        return [substance.moles for substance in self.mixture.substances]

    @property
    def volume(self):
        return self.solvent.volume

    def __repr__(self):
        return " and ".join([
        f"{self.concentrations[substance.species]} of {substance.species}"
        for substance in self.mixture.substances
        ]) + f" in {self.solvent}"

    def __add__(self, x):
        if isinstance(x, Solvent):
            assert self.solvent.species == x.species, "currently don't suppose mixed solvent solutions"
            return ComplexSolution(
                mixture = self.mixture,
                solvent = self.solvent + x
            )
        elif isinstance(x, Solution):
            assert self.solvent.species == x.solvent.species, "currently don't suppose mixed solvent solutions"
            return ComplexSolution(
                mixture = self.mixture + x.substances,
                solvent = self.solvent + x.solvent
            )
        elif isinstance(x, ComplexSolution):
            assert self.solvent.species == x.solvent.species, "currently don't suppose mixed solvent solutions"
            return ComplexSolution(
                mixture = self.mixture + x.mixture,
                solvent = self.solvent + x.solvent
            )
        else:
            raise NotImplementedError(f"adding between {type(self)} and {type(x)} not supported")

    def __mul__(self, x):
        assert isinstance(x, float)

        return ComplexSolution(
            mixture = x * self.mixture,
            solvent = x * self.solvent
        )

    # TODO: change to work
    def aliquot(self, volume):
        """ Split into aliquot and source """
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

        aliquot = ComplexSolution(
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

        source = ComplexSolution(
            mixture = source_mixture,
            solvent = source_solvent,
            concentrations = self.concentrations
        )

        return aliquot, source

class Container(Group):
    """ Simple container for a solution """
    def __init__(self, solution=None, **values):
        super(Container, self).__init__(solution = solution, **values)

    @property
    def volume(self):
        if self.solution is not None:
            return self.solution.volume
        else:
            return Quantity(0.0, VOLUME_UNIT)

    def retrieve_aliquot(self, volume):
        """ Removes an aliquot and returns it """
        assert self.solution is not None # TODO: check that volume is enough to remove an aliquot
        aliquot, source = self.solution.aliquot(volume)
        self.solution = source
        return aliquot

    def receive_aliquot(self, solution):
        if self.solution is not None:
            self.solution = self.solution + solution
        else:
            self.solution = solution

    def __repr__(self):
        return f"Well containing {self.solution}"

    def observe(self, value, key=None):
        if key:
            return getattr(self.solution, value)[key]
        else:
            return getattr(self.solution, value)

class WellArray(Container):
    """ An array of Containers (e.g. well plate). Must contain an array """
    def __init__(self, solution=None, **values):
        assert (isinstance(solution.volume.magnitude, np.ndarray) or
                isinstance(solution.volume.magnitude, jnp.ndarray))
        super(WellArray, self).__init__(solution=solution, **values)

    def retrieve_well_aliquot(self, idx, volume):
        raise NotImplementedError()
