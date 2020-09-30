# =============================================================================
# IMPORTS
# =============================================================================
import abc
import bellini
from collections import OrderedDict

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Story(abc.ABC):
    """ A procedure that involves groups, quantities, and distributions. """
    def __init__(self):
        super(Story, self).__init__()
        self.objects = OrderedDict()

    def register(self, name, x):
        self.objects[name] = x

    def __getattr__(self, name):
        return self.objects[name]
