# =============================================================================
# IMPORTS
# =============================================================================
import abc
import bellini
from bellini import Quantity, Distribution, Group
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

    def __setattr__(self, name, x):
        if isinstance(x, Group):
            self.objects[name] = x
        super(Story, self).__setattr__(name, x)

    def __getitem__(self, name):
        return self.objects[name]

    def __setitem__(self, name, x):
        self.objects[name] = x

    def __repr__(self):
        return 'Story containinig %s' % str(self.objects)