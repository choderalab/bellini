# =============================================================================
# IMPORTS
# =============================================================================
#import pint
import numpy as np
import jax.numpy as jnp
import torch
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
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, jnp.ndarray):
            return x
        if isinstance(x, torch.Tensor):
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
