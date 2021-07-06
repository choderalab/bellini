# =============================================================================
# IMPORTS
# =============================================================================
import abc
import bellini
from bellini.api import utils
from bellini.units import *
import bellini.api.functional as F
import numpy as np

# =============================================================================
# BASE CLASSES
# =============================================================================
class Distribution(abc.ABC):
    """ Base class for distributions. """
    def __init__(self, observed=False, name=None, **parameters):
        self.parameters = parameters
        self.observed = observed
        self._name = name
        for name, parameter in parameters.items():
            # assert isinstance(parameter, Quantity)
            setattr(self, name, parameter)

    @property
    def name(self):
        if self._name is not None:
            return self._name
        else:
            return self.__repr__()

    @name.setter
    def name(self, x):
        assert isinstance(x, str)
        self._name = x

    def _build_graph(self):
        import networkx as nx # local import
        g = nx.MultiDiGraph() # distribution always start with a fresh graph
        g.add_node(self, ntype='distribution')
        for name, parameter in self.parameters.items():

            if isinstance(parameter, Distribution):
                g.add_node(parameter, ntype='parameter')
                g.add_edge(
                    parameter,
                    self,
                    etype='is_parameter_of',
                )
                g = nx.compose(g, parameter.g)

            else:
                g.add_node(parameter, ntype='parameter', name=name)
                g.add_edge(
                    parameter,
                    self,
                    etype='is_parameter_of',
                )

        self._g = g
        return g

    @property
    def g(self):
        if not hasattr(self, '_g'):
            self._build_graph()
        return self._g

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

    @abc.abstractproperty
    def dimensionality(self):
        raise NotImplementedError

    @abc.abstractproperty
    def magnitude(self):
        raise NotImplementedError

    @abc.abstractproperty
    def units(self):
        raise NotImplementedError

    @property
    def shape(self):
        return self.magnitude.shape

    def __add__(self, x):
        return ComposedDistribution([self, x], op="add")

    def __radd__(self, x):
        return self.__add__(x)

    def __sub__(self, x):
        return ComposedDistribution([self, x], op="sub")

    def __rsub__(self, x):
        return ComposedDistribution([x, self], op="sub")

    def __neg__(self):
        return TransformedDistribution([self], op='neg')

    def __mul__(self, x):
        if isinstance(x, bellini.Species):
            # allows us to define Substances from Species
            # by multiplying by Distributions
            return NotImplemented
        return ComposedDistribution([self, x], op="mul")

    def __rmul__(self, x):
        return self.__mul__(x)

    def __truediv__(self, x):
        return ComposedDistribution([self, x], op="div")

    def __rtruediv__(self, x):
        return ComposedDistribution([x, self], op="div")

    def __pow__(self, x):
        return TransformedDistribution([self, x], op="power")

    def __rpow__(self, x):
        return TransformedDistribution([x, self], op="power")

    def __getitem__(self, idxs):
        parameters = self.parameters.copy()
        for name, param in parameters.items():
            if utils.is_arr(param):
                parameters[name] = param[idxs]
        instance = self.__class__(
            observed=self.observed,
            name=self.name,
            **parameters)
        return instance


class SimpleDistribution(Distribution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ComposedDistribution(Distribution):
    """ A composed distribution made of two distributions. """
    def __init__(self, distributions, op):
        super().__init__()
        assert len(distributions) == 2 # two at a time
        assert utils.check_shape(*distributions)
        self.distributions = distributions
        self.op = op

    @property
    def magnitude(self):
        return getattr(F, self.op)(
            self.distributions[0].magnitude,
            self.distributions[1].magnitude,
        )

    @property
    def dimensionality(self):
        try:
            with bellini.inference(False):
                np_args = [
                    bellini.Quantity(arg.magnitude, arg.units)
                    for arg in self.distributions
                ]
                return getattr(F, self.op)(
                    *np_args,
                ).dimensionality
        except ValueError:
            raise NotImplementedError("computing dimensionality for given operation not supported")

    @property
    def units(self):
        try:
            with bellini.inference(False):
                np_args = [
                    bellini.Quantity(arg.magnitude, arg.units)
                    for arg in self.distributions
                ]
                return getattr(F, self.op)(
                    *np_args,
                ).units
        except ValueError:
            raise NotImplementedError("computing dimensionality for given operation not supported")

    def _build_graph(self):
        import networkx as nx # local import
        g = nx.MultiDiGraph() # distribution always start with a fresh graph
        g.add_node(self, ntype='composed_distribution', op=self.op)
        g.add_node(self.distributions[0], ntype='first_distribution')
        g.add_node(self.distributions[1], ntype='second_distribution')
        g.add_edge(
            self.distributions[0],
            self,
            etype='is_first_distribution_of',
        )
        g.add_edge(
            self.distributions[1],
            self,
            etype='is_second_distribution_of',
        )
        g = nx.compose(g, self.distributions[0].g)
        g = nx.compose(g, self.distributions[1].g)
        self._g = g
        return g

    def __repr__(self):
        if bellini.verbose:
            return 'ComposedDistriubution: (%s %s %s)' % (
                repr(self.distributions[0]),
                self.op,
                repr(self.distributions[1]),
            )
        else:
            import numpy as np
            import jax.numpy as jnp
            if utils.is_arr(self.magnitude):
                mag = repr(self.magnitude)
            else:
                mag = f"{self.magnitude:.2f}"
            return f'CompDist w mag {mag} {self.units:~P}'

    def __getitem__(self, idxs):
        return ComposedDistribution(
            (
                self.distributions[0][idxs],
                self.distributions[1][idxs]
            ),
            op = self.op
        )


class TransformedDistribution(Distribution):
    """ A transformed distribution from one base distribution. """
    def __init__(self, args, op, **kwargs):
        super().__init__()
        args_contains_dist = np.array([
            isinstance(arg, Distribution)
            for arg in args
        ]).any()
        assert args_contains_dist, "TransformedDistribution must have an Distribution as an argument"

        self.args = args
        self.op = op
        self.kwargs = kwargs

    def _build_graph(self):
        import networkx as nx # local import
        g = nx.MultiDiGraph() # new graph
        g.add_node(
            self,
            ntype="transformed_distribution",
            op=self.op,
            kwargs = self.kwargs
        )
        for idx, arg in enumerate(self.args):
            g.add_node(
                arg,
                ntype="arg",
                pos=idx
            )
            g.add_edge(
                arg,
                self,
                etype="is_arg_of"
            )
            g = nx.compose(g, arg.g)
        self._g = g
        return g

    def __repr__(self):
        return 'TransformedDistribution: (%s %s with %s)' % (
            self.op,
            self.args,
            self.kwargs,
        )

    @property
    def magnitude(self):
        return getattr(F, self.op)(
            *[arg.magnitude for arg in self.args],
            **self.kwargs
        )

    @property
    def dimensionality(self):
        try:
            with bellini.inference(False):
                np_args = [
                    bellini.Quantity(arg.magnitude, arg.units)
                    for arg in self.args
                ]
                return getattr(F, self.op)(
                    *np_args,
                    **self.kwargs
                ).dimensionality
        except ValueError:
            raise NotImplementedError("computing dimensionality for given operation not supported")

    @property
    def units(self):
        try:
            with bellini.inference(False):
                np_args = [
                    bellini.Quantity(arg.magnitude, arg.units)
                    for arg in self.args
                ]
                return getattr(F, self.op)(
                    *np_args,
                    **self.kwargs
                ).units
        except ValueError:
            raise NotImplementedError("computing units for given operation not supported")

    def __getitem__(self, idxs):
        return TransformedDistribution(
            [self, idxs],
            op = "slice"
        )


# =============================================================================
# MODULE CLASSES
# =============================================================================
class Normal(SimpleDistribution):
    """ Normal distribution. """
    def __init__(self, loc, scale, **kwargs):
        assert loc.dimensionality == scale.dimensionality
        super().__init__(**kwargs)
        self.loc = loc
        self.scale = scale

    @property
    def dimensionality(self):
        return self.loc.dimensionality

    @property
    def magnitude(self):
        return self.loc.magnitude

    @property
    def units(self):
        return self.loc.units

    def __repr__(self):
        if bellini.verbose:
            return super().__repr__()
        else:
            if not isinstance(self.loc, Distribution):
                u = f'{self.loc:~P.2f}'
            else:
                u = f'{self.loc}'
            if not isinstance(self.scale, Distribution):
                sig2 = f'{self.scale**2:~P.2f}'
            else:
                sig2 = f'{self.scale**2}'

            return f'N({u}, {sig2})'


class Uniform(SimpleDistribution):
    """ Uniform distribution. """
    def __init__(self, low, high, **kwargs):
        assert low.dimensionality == high.dimensionality
        super().__init__(**kwargs)
        self.low = low
        self.high = high

    @property
    def dimensionality(self):
        return self.low.dimensionality

    @property
    def magnitude(self):
        return (self.high.magnitude - self.low.magnitude)/2

    @property
    def units(self):
        return self.low.units

    def __repr__(self):
        if bellini.verbose:
            return super().__repr__()
        else:
            if not isinstance(self.low, Distribution):
                low = f'{self.low:~P.2f}'
            else:
                low = f'{self.low}'
            if not isinstance(self.high, Distribution):
                high = f'{self.high:~P.2f}'
            else:
                high = f'{self.high}'

            return f'U({low}, {high})'
