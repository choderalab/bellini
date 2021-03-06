# =============================================================================
# IMPORTS
# =============================================================================
import abc
import bellini
from bellini import Quantity

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

    def __sub__(self, x):
        return ComposedDistribution([self, x], op="sub")

    def __mul__(self, x):
        return ComposedDistribution([self, x], op="mul")

    def __div__(self, x):
        return ComposedDistribution([self, x], op="div")

    def __pow__(self, x):
        return TransformedDistribution(self, op="pow", order=x)

    def __abs__(self):
        return TransformedDistribution(self, op="abs")

    def exp(self):
        # TODO: re-write math
        return TransformedDistribution(self, op="exp")

    def __exp__(self):
        return self.exp()

    def log(self):
        # TODO: re-write math
        return TransformedDistribution(self, op="exp")

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
        return self.distributions[0].dimensionality

    @property
    def units(self):
        return self.distributions[0].units

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
        return 'ComposedDistriubution: %s %s %s' % (
            self.distributions[0].name,
            self.op,
            self.distributions[1].name,
        )

class TransformedDistribution(Distribution):
    """ A transformed distribution from one base distribution. """
    def __init__(self, distribution, op, **kwargs):
        super(TransformedDistribution, self).__init__()
        self.distribution = distribution
        self.op = op
        for key, value in kwargs:
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
        return 'TransformedDistribution: %s %s with %s' % (
            self.op,
            self.distribution.name,
            self.kwargs,
        )

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
