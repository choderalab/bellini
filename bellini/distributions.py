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
    def __init__(self, **parameters):
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

class TransformedDistribution(Distribution):
    """ A transformed distribution from one base distribution. """
    def __init__(self, distribution, op, **kwargs):
        super(TransformedDistribution, self).__init__()
        self.distribution = distribution
        self.op = op
        for key, value in kwargs:
            setattr(self, key, value)

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Delta(SimpleDistribution):
    """ Degenerate discrete distribution. (A single point). """
    def __init__(self, x):
        self.x = x

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
