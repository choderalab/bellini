# =============================================================================
# IMPORTS
# =============================================================================

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Node(object):
    def __init__(self, parents=[], children={}, relations=[]):
        super(Node, self).__init__()
        self.parents = parents
        self.children = children
        self.relations = relations

    def __getattr__(self, name):
        # allow children to be accessed by `__getattr__`
        if name != "children":
            if name in self.children:
                return self.children[name]

        else:
            super(Node, self).__getattribute__(name)

    def _execute(self):
        for relation in self.relations:
            for key, value in relation().items():
                setattr(self, key, value)

    
