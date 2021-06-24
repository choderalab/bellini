# =============================================================================
# IMPORTS
# =============================================================================
#import pint
import numpy as np
import jax.numpy as jnp
import torch
import bellini.distributions as dist
from bellini.units import *
# =============================================================================
# MODULE CLASSES
# =============================================================================
class Quantity(pint.quantity.Quantity):
    """ A class that describes physical quantity, which contains
    numeric value and units.
    """

    @staticmethod
    def _convert_to_numpy(x):
        if isinstance(x, float):
            return x
        elif isinstance(x, np.ndarray):
            return x
        elif isinstance(x, jnp.ndarray):
            return x
        elif isinstance(x, torch.Tensor):
            # TODO:
            # do not require torch import ahead of time
            return x.numpy()
        raise ValueError("input could not be converted to numpy!")

    def __new__(self, value, unit, name=None):
        value = self._convert_to_numpy(value)
        if name is None:
            name = repr(self)
        self.name = name
        return super(Quantity, self).__new__(self, value, unit)

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
        if isinstance(x, dist.Distribution):
            return x + self
        else:
            return super(Quantity, self).__add__(x)

    def __sub__(self, x):
        if isinstance(x, dist.Distribution):
            return -x + self
        else:
            return super(Quantity, self).__sub__(x)

    def __mul__(self, x):
        if isinstance(x, dist.Distribution):
            return x * self
        else:
            return super(Quantity, self).__mul__(x)

    def __truediv__(self, x):
        if isinstance(x, dist.Distribution):
            return (x ** -1) * self
        else:
            return super(Quantity, self).__truediv__(x)

    def __hash__(self):
        self_base = self.to_base_units()
        # TODO: faster way to hash an array?
        # str(arr.sum()) + str((arr**2).sum()) is a possibility for large arrays
        if isinstance(self.magnitude, np.ndarray) or isinstance(self.magnitude, jnp.ndarray):
            return hash((self_base.__class__, self_base.magnitude.tobytes(), self_base.units))
        else:
            return super(Quantity, self).__hash__()

    def __eq__(self, other):
        iseq = super(Quantity, self).__eq__(other)
        if isinstance(iseq, np.ndarray) or isinstance(iseq, jnp.ndarray):
            return iseq.all()
