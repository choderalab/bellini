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

class ComposedDistribution(Distribution):
    """ A composed distribution made of two distributions. """
    def __init__(self, distributions, operator):
        super(ComposedDistribution, self).__init__()

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
