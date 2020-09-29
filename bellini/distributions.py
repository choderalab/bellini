# =============================================================================
# IMPORTS
# =============================================================================
import abc
import bellini

# =============================================================================
# BASE CLASSES
# =============================================================================
class Distribution(abc.ABC):
    """ Base class for distributions. """
    allowed_instances = (bellini.quantity.Quantity, )
    def __init__(self, **parameters):
        assert all(
            isinstance(parameter, self.allowed_instances)
            for _, parameter in parameters.items()
        ), "input instance is not allowed."

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Normal(Distribution):
    """ Normal distribution. """
