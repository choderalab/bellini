# =============================================================================
# IMPORTS
# =============================================================================
import abc
import bellini
from bellini import Quantity, Distribution, Group
from bellini.containers import Container
from bellini.devices import Device, ActionableDevice, MeasurementDevice
from collections import OrderedDict

# =============================================================================
# MODULE CLASSES
# =============================================================================
def _is_exp_obj(x):
    return isinstance(x, (
        Group, Distribution, Quantity, Container
    ))

class Procedure(abc.ABC):
    """
    A procedure that involves a series of experimental states
    (composed of Groups, Distributions, and Quantities),
    as well as instruments to manipulate these states
    """
    def __init__(self, objects=None, devices=None):
        super(Procedure, self).__init__()
        if objects:
            assert isinstance(objects, dict)
            self.exp_state = objects
        else:
            self.exp_state = {}

        if devices:
            assert isinstance(devices, dict)
            self.devices = devices
        else:
            self.devices = {}

        self.timeline = [self.exp_state]
        self.belief_subgraphs = []

    def register(self, name, x):
        if isinstance(x, Device):
            self.devices[name] = x
        elif _is_exp_obj(x):
            self.exp_state[name] = x
        else:
            raise ValueError("only experimental objects and devices can be registered")

    def perform(self, actionable_name, **arg_names):
        """
        `arg_names` should correspond to the device's `apply_state()` signature
        """
        # apply device to state to get new state
        device = self.devices[actionable_name]
        assert isinstance(device, ActionableDevice), f"cannot use `perform` with {device}"
        new_exp_state, belief_subgraph = device.apply_state(self.exp_state, **arg_names)
        # set most recent state to new state
        self.exp_state = new_exp_state
        self.timeline.append(new_exp_state)
        # update belief subgraphs
        self.belief_subgraphs.append((belief_subgraph, device))
        # an index for future exp_state retrieval if necessary
        return len(self.timeline)-1

    def measure(self, measurement_name, **arg_names):
        """
        `arg_names` should correspond to the device's `apply_state()` signature
        """
        device = self.devices[measurement_name]
        assert isinstance(device, MeasurementDevice), f"cannot use `measure` with {device}"
        results = device.readout_state(self.exp_state, **arg_names)
        return results

    def apply_law(self, law, container_name):
        # get container and apply law
        container = self.exp_state[container_name]
        lawed_container = container.apply_law(law)
        # generate new exp state and update timeline
        new_exp_state = self.exp_state.copy()
        new_exp_state[container_name] = lawed_container
        self.exp_state = new_exp_state
        self.timeline.append(new_exp_state)
        # update belief subgraphs
        belief_graph = {lawed_container: (container,)}
        self.belief_subgraphs.append((belief_graph, law))
        # index for future exp retrieval if necessary
        return len(self.timeline)-1

    def __getattr__(self, name):
        if name in self.exp_state:
            return self.exp_state[name]
        elif name in self.devices:
            return self.devices[name]
        else:
            super(Procedure, self).__getattribute__(name)

    def __setattr__(self, name, x):
        if isinstance(x, Device) or _is_exp_obj(x):
            self.register(name, x)
        else:
            super(Procedure, self).__setattr__(name, x)

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __setitem__(self, name, x):
        self.__setattr__(name, x)

    def __repr__(self):
        return 'Procedure containing %s' % str(self.objects)

    def _build_graph(self):
        import networkx as nx # local import
        g = nx.MultiDiGraph() # start with a fresh graph

        # add relations via subgraphs
        for idx, (subgraph, device) in enumerate(self.belief_subgraphs):
            for child, parents in subgraph.items():
                for parent in parents:
                    g.add_node(
                        parent,
                        layer=idx
                    )
                    g.add_node(
                        child,
                        layer=idx+1
                    )
                    g.add_edge(
                        parent,
                        child,
                        device=device.name
                    )

        self._g = g
        return g

    @property
    def g(self):
        if not hasattr(self, '_g'):
            self._build_graph()
        return self._g
