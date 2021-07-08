# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import jax.numpy as jnp
import torch
import bellini
from bellini.units import *
from bellini.api import utils
from pint.errors import DimensionalityError

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Quantity(pint.quantity.Quantity):
    """ A class that describes physical quantity, which contains
    numeric value and units.
    """

    @staticmethod
    def _convert_to_numpy(x):
        if isinstance(x, (float, int)):
            return np.array(x)
        elif isinstance(x, (np.generic, np.ndarray)):
            return x
        elif isinstance(x, jnp.ndarray):
            return np.array(x)
        elif isinstance(x, torch.Tensor):
            # TODO: do not require torch import ahead of time
            return x.numpy()
        print(type(x), x)
        raise ValueError("input could not be converted to numpy!")

    @staticmethod
    def _convert_to_jnp(x):
        if isinstance(x, (float, int)):
            return jnp.array(x)
        elif isinstance(x, (np.generic, np.ndarray)):
            return jnp.array(x)
        elif isinstance(x, jnp.ndarray):
            return x
        elif isinstance(x, torch.Tensor):
            # TODO: do not require torch import ahead of time
            return jnp.array(x)
        raise ValueError("input could not be converted to jnp!")

    def __new__(cls, value, unit=None, name=None):
        if bellini.infer:
            value = cls._convert_to_jnp(value)
        else:
            value = cls._convert_to_numpy(value)
        if name is None:
            name = repr(cls)
        cls.name = name
        return super().__new__(cls, value, unit)

    def jnp(self):
        r""" return self but jnp.ndarray """
        value = self._convert_to_jnp(self.magnitude)
        unit = self.units
        instance = self.__new__(self.__class__, value, unit)
        instance.name = self.name
        return instance

    def unitless(self):
        r""" return self but unitless """
        instance = self.__new__(self.__class__, self.magnitude)
        instance.name = self.name
        #print("unitless", type(instance))
        return instance

    def to_units(self, new_units, force=False):
        try:
            return self.to(new_units)
        except DimensionalityError as e:
            if not force:
                print(f"cannot convert {self.units} to {new_units}. if you'd like to assign new units, use force=True")
                raise e
            instance = self.__new__(self.__class__, self.magnitude, new_units)
            instance.name = self.name
            return instance

    def _build_graph(self):
        import networkx as nx
        g = nx.MultiDiGraph()
        g.add_node(self, ntype='quantity', name=self.name)
        self._g = g
        return g

    @property
    def g(self):
        if not hasattr(self, '_g'):
            self._build_graph()
        return self._g

    def __add__(self, x):
        if isinstance(x, (bellini.Distribution, bellini.Group)):
            return NotImplemented
        return super().__add__(x)

    def __sub__(self, x):
        if isinstance(x, (bellini.Distribution, bellini.Group)):
            return NotImplemented
        return super().__sub__(x)

    def __mul__(self, x):
        if isinstance(x, (bellini.Distribution, bellini.Group)):
            return NotImplemented
        return super().__mul__(x)

    def __truediv__(self, x):
        if isinstance(x, (bellini.Distribution, bellini.Group)):
            return NotImplemented
        return super().__truediv__(x)

    def __pow__(self, x):
        if isinstance(x, (bellini.Distribution, bellini.Group)):
            return NotImplemented
        return super().__pow__(x)

    def __setitem__(self, key, value):
        if isinstance(self.magnitude, jnp.ndarray):
            new_value = self.magnitude.at[key].set(value)
            # ? this breaks the "functional" concept
            # but is consistent with np syntax
            self.magnitude = new_value
        else:
            super().__setitem__(key, value)

    def __hash__(self):
        self_base = self.to_base_units()
        # TODO: faster way to hash an array?
        # str(arr.sum()) + str((arr**2).sum()) is a possibility for large arrays
        if utils.is_arr(self.magnitude):
            return hash((self_base.__class__, self_base.magnitude.tobytes(), self_base.units))
        return super().__hash__()

    def __eq__(self, other):
        is_eq = super().__eq__(other)
        if utils.is_arr(is_eq):
            return is_eq.all()
        return is_eq
