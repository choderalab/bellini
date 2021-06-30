# =============================================================================
# IMPORTS
# =============================================================================
import abc
import bellini
from bellini.units import *

# =============================================================================
# CONSTANTS
# =============================================================================
OPS = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y,
    "exp": lambda x: x.exp(),
    "log": lambda x: x.log(),
    "neg": lambda x: -x,
    "pow": lambda x, y: x ** y,
}

# =============================================================================
# BASE CLASSES
# =============================================================================
class Distribution(abc.ABC):
    """ Base class for distributions. """
    def __init__(self, observed=False, name=None, **parameters):
        self.parameters = parameters
        self.observed = observed
        self._name = name
        for name, parameter in parameters.items():
            # assert isinstance(parameter, Quantity)
            setattr(self, name, parameter)

    @property
    def name(self):
        if self._name is not None:
            return self._name
        else:
            return self.__repr__()

    @name.setter
    def name(self, x):
        assert isinstance(x, str)
        self._name = x

    def _build_graph(self):
        import networkx as nx # local import
        g = nx.MultiDiGraph() # distribution always start with a fresh graph
        g.add_node(self, ntype='distribution')
        for name, parameter in self.parameters.items():

            if isinstance(parameter, Distribution):
                g.add_node(parameter, ntype='parameter')
                g.add_edge(
                    parameter,
                    self,
                    etype='is_parameter_of',
                )
                g = nx.compose(g, parameter.g)

            else:
                g.add_node(parameter, ntype='parameter', name=name)
                g.add_edge(
                    parameter,
                    self,
                    etype='is_parameter_of',
                )

        self._g = g
        return g

    @property
    def g(self):
        if not hasattr(self, '_g'):
            self._build_graph()
        return self._g

    def __repr__(self):
        return '%s distribution with %s' % (
            self.__class__.__name__,
            ' and '.join(
                [
                    '%s=%s' % (
                        name,
                        parameter
                    ) for name, parameter in self.parameters.items()
                ]
            )
        )

    @abc.abstractproperty
    def dimensionality(self):
        raise NotImplementedError

    @abc.abstractproperty
    def magnitude(self):
        raise NotImplementedError

    @abc.abstractproperty
    def units(self):
        raise NotImplementedError

    def __add__(self, x):
        return ComposedDistribution([self, x], op="add")

    def __radd__(self, x):
        return self.__add__(x)

    def __sub__(self, x):
        return ComposedDistribution([self, x], op="sub")

    def __rsub__(self, x):
        return ComposedDistribution([x, self], op="sub")

    def __neg__(self):
        return TransformedDistribution(self, op='neg')

    def __mul__(self, x):
        return ComposedDistribution([self, x], op="mul")

    def __rmul__(self, x):
        return self.__mul__(x)

    def __truediv__(self, x):
        return ComposedDistribution([self, x], op="div")

    def __rtruediv__(self, x):
        return ComposedDistribution([x, self], op="div")

    def __pow__(self, x):
        return TransformedDistribution(self, op="pow", order=x)

    def __rpow__(self, x):
        return TransformedDistribution(x, op="pow", order=self)

    def __abs__(self):
        return TransformedDistribution(self, op="abs")

    def exp(self):
        # TODO: re-write math
        return TransformedDistribution(self, op="exp")

    def __exp__(self):
        return self.exp()

    def log(self):
        # TODO: re-write math
        return TransformedDistribution(self, op="log")

    def __log__(self):
        return self.log()


class SimpleDistribution(Distribution):
    def __init__(self, *args, **kwargs):
        super(SimpleDistribution, self).__init__(**kwargs)


class ComposedDistribution(Distribution):
    """ A composed distribution made of two distributions. """
    def __init__(self, distributions, op):
        super(ComposedDistribution, self).__init__()
        assert len(distributions) == 2 # two at a time
        self.distributions = distributions
        self.op = op

    @property
    def magnitude(self):
        return OPS[self.op](
            self.distributions[0].magnitude,
            self.distributions[1].magnitude,
        )

    @property
    def dimensionality(self):
        if self.op == "add" or self.op == "sub":
            return self.distributions[0].dimensionality
        elif self.op == "mul":
            return self.distributions[0].dimensionality * self.distributions[1].dimensionality
        elif self.op == "div":
            return self.distributions[0].dimensionality / self.distributions[1].dimensionality
        else:
            raise Exception("cannot compute dimensionality for given operation")

    @property
    def units(self):
        if self.op == "add" or self.op == "sub":
            return self.distributions[0].units
        elif self.op == "mul":
            return self.distributions[0].units * self.distributions[1].units
        elif self.op == "div":
            return self.distributions[0].units / self.distributions[1].units
        else:
            raise Exception("cannot compute units for given operation")

    def _build_graph(self):
        import networkx as nx # local import
        g = nx.MultiDiGraph() # distribution always start with a fresh graph
        g.add_node(self, ntype='composed_distribution', op=self.op)
        g.add_node(self.distributions[0], ntype='first_distribution')
        g.add_node(self.distributions[1], ntype='second_distribution')
        g.add_edge(
            self.distributions[0],
            self,
            etype='is_first_distribution_of',
        )
        g.add_edge(
            self.distributions[1],
            self,
            etype='is_second_distribution_of',
        )
        g = nx.compose(g, self.distributions[0].g)
        g = nx.compose(g, self.distributions[1].g)
        self._g = g
        return g

    def __repr__(self):
        if bellini.verbose:
            return 'ComposedDistriubution: (%s %s %s)' % (
                repr(self.distributions[0]),
                self.op,
                repr(self.distributions[1]),
            )
        else:
            import numpy as np
            import jax.numpy as jnp
            if isinstance(self.magnitude, np.ndarray) or isinstance(self.magnitude, jnp.ndarray):
                mag = repr(self.magnitude)
            else:
                mag = f"{self.magnitude:.2f}"
            return f'CompDist w mag {mag} {self.units:~P}'


class TransformedDistribution(Distribution):
    """ A transformed distribution from one base distribution. """
    def __init__(self, distribution, op, **kwargs):
        super(TransformedDistribution, self).__init__()
        self.distribution = distribution
        self.op = op
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _build_graph(self):
        import networkx as nx # local import
        g = self.distribution.g.copy() # copy original graph
        g.add_node(
            self,
            ntype="transformed_distribution",
            op=self.op,
        )
        g.add_edge(
            self.distribution,
            self,
            etype="is_base_distribution_of"
        )
        self._g = g
        return g

    def __repr__(self):
        return 'TransformedDistribution: (%s %s with %s)' % (
            self.op,
            self.distribution.name,
            self.kwargs,
        )

    @property
    def magnitude(self):
        return OPS[self.op](
            self.distribution
        )

    @property
    def dimensionality(self):
        if self.op == "neg":
            return self.distribution.dimensionality
        elif self.op == "pow":
            return self.distribution.dimensionality ** self.order
        else:
            raise NotImplementedError("computing dimensionality for given operation not supported")

    @property
    def units(self):
        if self.op == "neg":
            return self.distribution.units
        elif self.op == "pow":
            return self.distribution.units ** self.order
        elif self.op == "log" or self.op == "exp":
            return ureg.dimensionless
        else:
            raise NotImplementedError("computing units for given operation not supported")


# =============================================================================
# MODULE CLASSES
# =============================================================================
class Normal(SimpleDistribution):
    """ Normal distribution. """
    def __init__(self, loc, scale, **kwargs):
        assert loc.dimensionality == scale.dimensionality
        super(Normal, self).__init__(loc=loc, scale=scale, **kwargs)

    @property
    def dimensionality(self):
        return self.loc.dimensionality

    @property
    def magnitude(self):
        return self.loc.magnitude

    @property
    def units(self):
        return self.loc.units

    def __repr__(self):
        if bellini.verbose:
            return super(Normal, self).__repr__()
        else:
            if not isinstance(self.loc, Distribution):
                u = f'{self.loc:~P.2f}'
            else:
                u = f'{self.loc}'
            if not isinstance(self.scale, Distribution):
                sig2 = f'{self.scale**2:~P.2f}'
            else:
                sig2 = f'{self.scale**2}'

            return f'N({u}, {sig2})'


class Uniform(SimpleDistribution):
    """ Uniform distribution. """
    def __init__(self, low, high, **kwargs):
        assert low.dimensionality == high.dimensionality
        super(Uniform, self).__init__(low=low, high=high, **kwargs)

    @property
    def dimensionality(self):
        return self.low.dimensionality

    @property
    def magnitude(self):
        return (self.high.magnitude - self.low.magnitude)/2

    @property
    def units(self):
        return self.low.units

    def __repr__(self):
        if bellini.verbose:
            return super(Uniform, self).__repr__()
        else:
            if not isinstance(self.low, Distribution):
                low = f'{self.low:~P.2f}'
            else:
                low = f'{self.low}'
            if not isinstance(self.high, Distribution):
                high = f'{self.high:~P.2f}'
            else:
                high = f'{self.high}'

            return f'U({low}, {high})'
