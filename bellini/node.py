# =============================================================================
# IMPORTS
# =============================================================================
import abc
import bellini

# =============================================================================
# BASE CLASSES
# =============================================================================
class NodeBase(abc.ABC):
    def __init__(self):
        super(NodeBase, self).__init__()

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Node(NodeBase):
    def __init__(self):
        super(Node, self).__init__()
        self.children = {}
        self.relations = []

    def __getattr__(self, name):
        # allow children to be accessed by `__getattr__`
        if name in self.children:
            return self.children[name]

        else:
            super(Node, self).__getattribute__(name)

    
