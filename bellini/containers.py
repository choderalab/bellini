# =============================================================================
# IMPORTS
# =============================================================================
import abc
from bellini.units import VOLUME_UNIT
from .quantity import Quantity
from bellini.api import utils
from .groups import Liquid

# =============================================================================
# Containers
# =============================================================================

class Container(object):
    """ Simple container for a solution """
    def __init__(self, solution=None, name=None, **values):
        if solution is not None:
            assert isinstance(solution, Liquid)
        self.solution = solution
        self.values = values
        self._name = name

    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return repr(self)

    def __getattr__(self, name):
        if name in self.values.keys():
            return self.values[name]
        else:
            return super().__getattribute__(name)

    def __eq__(self, new_group):
        return {
                **self.values,
                'name': self.name
            } ==  {
                **new_group.values,
                'name': new_group.name
            }

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
        return aliquot, Container(solution=source, name=self.name)

    def receive_aliquot(self, solution):
        if self.solution is not None:
            new_solution = self.solution + solution
        else:
            new_solution = solution
        return Container(solution=new_solution, name=self.name)

    def apply_law(self, law):
        if self.solution:
            return Container(
                law(self.solution)
            )
        else:
            raise ValueError("container has no solution to apply law to!")

    def __repr__(self):
        return f"Well containing {self.solution}"

    def observe(self, value, key=None):
        if key:
            return getattr(self.solution, value)[key]
        else:
            return getattr(self.solution, value)

    def __hash__(self):
        return hash(id(self))
