"""
Module containing various containers, which store Groups and interface with
Procedure
"""
# =============================================================================
# IMPORTS
# =============================================================================
import abc
from bellini.units import VOLUME_UNIT
from bellini.quantity import Quantity
from bellini.api import utils
from bellini.groups import Liquid
from bellini.reference import Reference as Ref

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

    @name.setter
    def name(self, x):
        assert isinstance(x, str)
        self._name = x

    def __getattr__(self, name):
        if name in self.values.keys():
            return self.values[name]
        else:
            return super().__getattribute__(name)

    def __eq__(self, new_group):
        return {
                **self.values,
                'name': self.name
            } == {
                **new_group.values,
                'name': new_group.name
            }

    @property
    def volume(self):
        """ The volume of the current solution """
        if self.solution is not None:
            return self.solution.volume
        else:
            return Quantity(0.0, VOLUME_UNIT)

    def retrieve_aliquot(self, volume):
        """ Removes an aliquot and returns it as well as the new Container with
        the aliquot removed """
        # TODO: check that volume is enough to remove an aliquot
        assert self.solution is not None
        aliquot, source = self.solution.aliquot(volume)
        return aliquot, Container(solution=source, name=self.name)

    def receive_aliquot(self, solution):
        """ Recieve an aliquot and return the new Container with aliquot added
        """
        if self.solution is not None:
            new_solution = self.solution + solution
        else:
            new_solution = solution
        return Container(solution=new_solution, name=self.name)

    def apply_law(self, law):
        """ Apply `law` to the current Solution and return it in a
        new Container """
        if self.solution:
            return Container(
                law(self.solution)
            )
        else:
            raise ValueError("container has no solution to apply law to!")

    def __repr__(self):
        return f"Well containing {self.solution}"

    def observe(self, value, key=None):
        """ Observe a particular attribute of the contained Solution """
        if isinstance(value, Ref):
            attr = getattr(self.solution, value.name)
            return value.retrieve_index(attr)
        else:
            if key:
                return getattr(self.solution, value)[key]
            else:
                return getattr(self.solution, value)

    def __hash__(self):
        return hash(id(self))
