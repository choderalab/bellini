# =============================================================================
# IMPORTS
# =============================================================================
import abc
import bellini
from bellini import Quantity

# =============================================================================
# BASE CLASSES
# =============================================================================
class Distribution(abc.ABC):
    """ Base class for distributions. """
    def __init__(self, **parameters):
        self.parameters = parameters
        for name, parameter in parameters.items():
            assert isinstance(parameter, Quantity)
            setattr(self, name, parameter)

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

    @property
    def dimensionality(self):
        raise NotImplementedError

    @property
    def magnitude(self):
        raise NotImplementedError

    def __add__(self, x):
        return ComposedDistribution([self, x], operator="add")

    def __sub__(self, x):
        return ComposedDistribution([self, x], operator="sub")

    def __mul__(self, x):
        return ComposedDistribution([self, x], operator="mul")

    def __div__(self, x):
        return ComposedDistribution([self, x], operator="div")

    def __pow__(self, x):
        return TransformedDistribution(self, operator="pow", order=x)

    def __abs__(self):
        return TransformedDistribution(self, operator="abs")

    def exp(self):
        # TODO: re-write math
        return TransformedDistribution(self, operator="exp")

    def __exp__(self):
        return self.exp()

    def log(self):
        # TODO: re-write math
        return TransformedDistribution(self, operator="exp")

    def __log__(self):
        return self.log()

class ComposedDistribution(Distribution):
    """ A composed distribution made of two distributions. """
    def __init__(self, distributions, operator):
        super(ComposedDistribution, self).__init__()
        self.distributions = distributions
        self.operator = operator

class TransformedDistribution(Distribution):
    """ A transformed distribution from one base distribution. """
    def __init__(self, distribution, operator, **kwargs):
        super(TransformedDistribution, self).__init__()
        self.distribution = distribution
        self.operator = operator
        for key, value in kwargs:
            setattr(self, key, value)


# =============================================================================
# MODULE CLASSES
# =============================================================================
class Normal(Distribution):
    """ Normal distribution. """
    def __init__(self, loc, scale):
        assert loc.dimensionality ** 0.5 == scale.dimensionality
        super(Normal, self).__init__(loc=loc, scale=scale)

    @property
    def dimensionality(self):
        return self.loc.dimensionality

    @property
    def magnitude(self):
        return self.loc.magnitude
