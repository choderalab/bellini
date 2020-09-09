# =============================================================================
# IMPORTS
# =============================================================================
import abc
import bellini as be

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Group(dict, abc.ABC):
    """ A group that holds quantities and distributions. """

    connections = []
    allow_children = True
    allowed_units = None

    def __setitem__(self, key, value):
        # assert value is either `Group` or `Quantity`
        assert (
                isinstance(value, be.quantity.Quantity)
                or
                isinstance(value, be.group.Group)
            ), "Group can only contain Quantity and Group."

        if self.allow_children is False and isinstance(
            value, be.group.Group
        ): raise RuntimeError('Attached Children when allow_children is Flase')

        if self.allowed_units is not None:
            assert any(
                allowed_unit.is_compatible_with(value.unit)
                for allowed_unit in self.allowed_units
            ), "Units not allowed."

        super().__setitem__(key, value)

    def connect(self, fn):
        """ Connect the nodes in the group. """
        assert callable(fn), "Edge has to be function."
        self.connections.append(fn)

    
