# =============================================================================
# IMPORTS
# =============================================================================
import abc
from .node import Node
from .quantity import Quantity

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
class Distribution(Node):
    """ Base class for distributions. """
    def __init__(self, **parameters):
        super(Distribution, self).__init__()
        for key, value in parameters.items():
            self.key = value

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
        return TransformedDistribution(self, op="exp")

    def log(self):
        return TransformedDistribution(self, op="log")

class SimpleDistribution(Distribution):
    def __init__(self, **kwargs):
        super(SimpleDistribution, self).__init__()
        self.kwargs = kwargs
        for key, value in kwargs.items():
            self.key = value

    def __repr__(self):
        return "%s with %s" % (
            self.__class__.__name__,
            self.kwargs
        )

class ComposedDistribution(Distribution):
    """ A composed distribution made of two distributions. """
    def __init__(self, distributions, op):
        super(ComposedDistribution, self).__init__()
        assert len(distributions) == 2 # two at a time
        self.distributions = distributions
        self.op = op
        self.children = {
            "child0": distributions[0],
            "child1": distributions[1]
        }
        self.relations = [
            lambda node: {"parent": OPS[self.op](
                node.child0, node.child1,
            )}
        ]

    def __repr__(self):
        return repr(self.distribution[0]) + self.op + repr(self.distributions[1])

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
    def unit(self):
        return self.distributions[0].unit

class TransformedDistribution(Distribution):
    """ A transformed distribution from one base distribution. """
    def __init__(self, distribution, op):
        super(TransformedDistribution, self).__init__()
        self.distribution = distribution
        self.op = op
        self.children = {
            "child0": distribution
        }
        self.relations = [
            lambda node: {"parent": OPS[self.op](node["child0"])}
        ]

    def __repr__(self):
        return self.op + str(self.distribution)

    @property
    def magnitude(self):
        return OPS[self.op](
            self.distribution.magnitude,
        )

    @property
    def dimensionality(self):
        return OPS[self.op](
            self.distribution.dimensionality
        )

    @property
    def unit(self):
        return OPS[self.op](
            self.distribution.unit
        )


# =============================================================================
# MODULE CLASSES
# =============================================================================
class Delta(SimpleDistribution):
    """ Degenerate discrete distribution. (A single point). """
    def __init__(self, x):
        super(Delta, self).__init__(x=x)

    @property
    def dimensionality(self):
        return self.x.dimensionality

    @property
    def magnitude(self):
        return self.x.magnitude

    @property
    def unit(self):
        return self.x.unit

class Normal(SimpleDistribution):
    """ Normal distribution. """
    def __init__(self, loc, scale):
        assert loc.dimensionality == scale.dimensionality
        super(Normal, self).__init__(loc=loc, scale=scale)

    @property
    def dimensionality(self):
        return self.loc.dimensionality

    @property
    def magnitude(self):
        return self.loc.magnitude

    @property
    def units(self):
        return self.loc.units
