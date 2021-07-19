# =============================================================================
# IMPORTS
# =============================================================================
import abc
import bellini
from bellini.api import utils
from bellini.units import ureg, to_internal_units, get_internal_units
import bellini.api.functional as F
import numpy as np
import warnings
from pint.errors import DimensionalityError

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

    @abc.abstractmethod
    def unitless(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_units(self, new_units, force=False):
        raise NotImplementedError()

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

    @abc.abstractproperty
    def internal_units(self):
        raise NotImplementedError

    @property
    def shape(self):
        return self.magnitude.shape

    def __add__(self, x):
        if utils.is_scalar(x):
            x = bellini.Quantity(x)
        return ComposedDistribution([self, x], op="add")

    def __radd__(self, x):
        return self.__add__(x)

    def __sub__(self, x):
        if utils.is_scalar(x):
            x = bellini.Quantity(x)
        return ComposedDistribution([self, x], op="sub")

    def __rsub__(self, x):
        if utils.is_scalar(x):
            x = bellini.Quantity(x)
        return ComposedDistribution([x, self], op="sub")

    def __neg__(self):
        return TransformedDistribution([self], op='neg')

    def __mul__(self, x):
        if isinstance(x, bellini.Species):
            # allows us to define Substances from Species
            # by multiplying by Distributions
            return NotImplemented
        elif utils.is_scalar(x):
            x = bellini.Quantity(x)
        return ComposedDistribution([self, x], op="mul")

    def __rmul__(self, x):
        return self.__mul__(x)

    def __truediv__(self, x):
        if utils.is_scalar(x):
            x = bellini.Quantity(x)
        return ComposedDistribution([self, x], op="div")

    def __rtruediv__(self, x):
        if utils.is_scalar(x):
            x = bellini.Quantity(x)
        return ComposedDistribution([x, self], op="div")

    def __pow__(self, x):
        if utils.is_scalar(x):
            x = bellini.Quantity(x)
        return TransformedDistribution([self, x], op="power")

    def __rpow__(self, x):
        if utils.is_scalar(x):
            x = bellini.Quantity(x)
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

        try:
            with bellini.inference(False):
                np_args = [
                    bellini.Quantity(arg.magnitude, arg.units)
                    if not isinstance(arg, bellini.Quantity) else arg
                    for arg in self.distributions
                ]
                mag = getattr(F, self.op)(
                    *np_args,
                )
                self._units = mag.units
                self._internal_units = get_internal_units(mag)
                self._magnitude = mag.magnitude
        except ValueError:
            raise NotImplementedError("computing units/dimensionality for given operation not supported")

    def unitless(self):
        return self.to_units(
            ureg.dimensionless,
            force=True
        )

    def to_units(self, new_units, force=False):
        return UnitChangedDistribution(
            self,
            new_units,
            force=force
        )

    @property
    def dimensionality(self):
        return self.units.dimensionality

    @property
    def magnitude(self):
        return self._magnitude

    @property
    def units(self):
        return self._units

    @property
    def internal_units(self):
        return self._internal_units

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
            return 'ComposedDistriubution: (%s %s %s)[%s]' % (
                self.distributions[0],
                self.op,
                self.distributions[1],
                self.units
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
        assert args_contains_dist, "TransformedDistribution must have a Distribution as an argument"

        self.args = args
        self.op = op
        self.kwargs = kwargs

        try:
            with bellini.inference(False):
                np_args = [
                    bellini.Quantity(arg.magnitude, arg.units)
                    if not isinstance(arg, bellini.Quantity) else arg
                    for arg in self.args
                ]
                mag = getattr(F, self.op)(
                    *np_args,
                    **self.kwargs
                )
                self._magnitude = mag.magnitude
                self._units = mag.units
                self._internal_units = get_internal_units(mag)
        except ValueError:
            raise NotImplementedError("computing magnitude or units for given operation not supported")

    def unitless(self):
        return self.to_units(
            ureg.dimensionless,
            force=True
        )

    def to_units(self, new_units, force=False):
        return UnitChangedDistribution(
            self,
            new_units,
            force=force
        )

    @property
    def dimensionality(self):
        return self.units.dimensionality

    @property
    def magnitude(self):
        return self._magnitude

    @property
    def units(self):
        return self._units

    @property
    def internal_units(self):
        return self._internal_units

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
        return 'TransformedDistribution: (%s %s with %s)[%s]' % (
            self.op,
            self.args,#[arg.unitless() for arg in self.args],
            self.kwargs,
            self.units
        )

    def __getitem__(self, idxs):
        return TransformedDistribution(
            [self, idxs],
            op = "slice"
        )


# TODO: i need a better name for this
class _JITDistribution(Distribution):
    """ A wrapper for functions of random variables with multiple outputs """
    def __init__(self, fn, inputs, label):
        super().__init__()
        self.fn = fn
        self.inputs = inputs
        self.label = label

        def to_quantity(arg):
            if isinstance(arg, bellini.Quantity):
                return arg
            else:
                if isinstance(arg, (list, tuple)):
                    return [to_quantity(r) for r in arg]
                elif isinstance(arg, dict):
                    return {key: to_quantity(value) for key, value in arg.items()}
                else:
                    return bellini.Quantity(arg.magnitude, arg.units)

        deterministic_args = {}
        for key, arg in self.inputs.items():
            deterministic_args[key] = to_quantity(arg)

        outputs = self.fn(**deterministic_args)
        self._magnitude = outputs[label].magnitude
        self._units = outputs[label].units
        self._internal_units = get_internal_units(outputs[label])

    @property
    def dimensionality(self):
        return self.units.dimensionality

    @property
    def magnitude(self):
        return self._magnitude

    @property
    def units(self):
        return self._units

    @property
    def internal_units(self):
        return self._internal_units

    def __repr__(self):
        if bellini.verbose:
            return '_JITDistribution: (%s with inputs %s, label %s)[%s]' % (
                self.fn,
                self.inputs,
                self.label,
                self.units
            )
        else:
            return f'_JTDist: {self.fn.__name__} for {self.label} [{self.units}]'

    def _build_graph(self):
        import networkx as nx # local import
        g = nx.MultiDiGraph() # new graph
        g.add_node(
            self,
            ntype="_jit_distribution",
        )
        for key, arg in self.inputs.items():
            g.add_node(
                arg,
                ntype="arg",
                key=key
            )
            g.add_edge(
                arg,
                self,
                etype="is_arg_of"
            )
            g = nx.compose(g, arg.g)
        self._g = g
        return g

    def unitless(self):
        return self.to_units(
            ureg.dimensionless,
            force=True
        )

    def to_units(self, new_units, force=False):
        return UnitChangedDistribution(
            self,
            new_units,
            force=force
        )


class UnitChangedDistribution(Distribution):
    def __init__(self, distribution, new_units, force=False):
        assert not isinstance(distribution, UnitChangedDistribution), "can't have nested UnitChangedDistributions"
        self.distribution = distribution
        self._units = new_units
        self._old_units = distribution.units

        # compute scaling factor
        scaling_factor = bellini.Quantity(1, (new_units / distribution.units)).to_base_units()
        if scaling_factor.units.dimensionality == ureg.dimensionless.dimensionality:
            self.scaling_factor = scaling_factor
        else:
            if force:
                warnings.warn("scaling factor can't be computed. assuming the scaling factor is 1")
                self.scaling_factor = 1
            else:
                raise DimensionalityError(self._old_units, self._units)

        self._magnitude = self.scaling_factor * distribution.magnitude
        self._internal_units = get_internal_units(new_units)

    @property
    def dimensionality(self):
        return self.units.dimensionality

    @property
    def magnitude(self):
        return self._magnitude

    @property
    def units(self):
        return self._units

    @property
    def internal_units(self):
        return self._internal_units

    def unitless(self):
        return UnitChangedDistribution(
            self.distribution,
            ureg.dimensionless,
            force=True
        )

    def to_units(self, new_units, force=False):
        return UnitChangedDistribution(
            self.distribution,
            new_units,
            force=force
        )


    def _build_graph(self):
        import networkx as nx # local import
        g = nx.MultiDiGraph() # distribution always start with a fresh graph
        g.add_node(self, ntype='unit_changed_distribution')
        g.add_node(self.distribution, ntype='base_distribution')
        g.add_edge(
            self.distribution,
            self,
            etype='is_base_distribution_of',
        )
        g = nx.compose(g, self.distribution.g)
        self._g = g
        return g

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Normal(SimpleDistribution):
    """ Normal distribution. """
    def __init__(self, loc, scale, **kwargs):
        assert loc.dimensionality == scale.dimensionality
        super().__init__(loc=loc, scale=scale, **kwargs)

    def unitless(self):
        return Normal(
            loc = to_internal_units(self.loc).unitless(),
            scale = to_internal_units(self.scale).unitless()
        )

    def to_units(self, new_units, force=False):
        try:
            return Normal(
                loc = self.loc.to_units(new_units, force=force),
                scale = self.scale.to_units(new_units, force=force)
            )
        except DimensionalityError as e:
            print(f"cannot convert {self.units} to {new_units}. if you'd like to assign new units, use force=True", file=sys.stderr)
            raise e

    @property
    def dimensionality(self):
        return self.loc.dimensionality

    @property
    def magnitude(self):
        return self.loc.magnitude

    @property
    def units(self):
        return self.loc.units

    @property
    def internal_units(self):
        return get_internal_units(self.loc)

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

            return f'N({u}, {sig2})[{self.units}]'


class Uniform(SimpleDistribution):
    """ Uniform distribution. """
    def __init__(self, low, high, **kwargs):
        assert low.dimensionality == high.dimensionality
        super().__init__(low=low, high=high, **kwargs)

    def unitless(self):
        return Uniform(
            low = to_internal_units(self.low).unitless(),
            high = to_internal_units(self.high).unitless()
        )

    def to_units(self, new_units, force=False):
        try:
            return Uniform(
                low = self.low.to_units(new_units, force=force),
                high = self.high.to_units(new_units, force=force)
            )
        except DimensionalityError as e:
            print(f"cannot convert {self.units} to {new_units}. if you'd like to assign new units, use force=True", file=sys.stderr)
            raise e

    @property
    def dimensionality(self):
        return self.low.dimensionality

    @property
    def magnitude(self):
        return (self.high.magnitude + self.low.magnitude)/2

    @property
    def units(self):
        return self.low.units

    @property
    def internal_units(self):
        return get_internal_units(self.low)

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

            return f'U({low}, {high})[{self.units}]'


class LogNormal(SimpleDistribution):
    """ LogNormal distribution. """
    def __init__(self, loc, scale, **kwargs):
        assert loc.dimensionality == scale.dimensionality
        super().__init__(loc=loc, scale=scale, **kwargs)

    def unitless(self):
        return LogNormal(
            loc = self.loc.unitless(),
            scale = self.scale.unitless()
        )

    def to_units(self, new_units, force=False):
        try:
            return LogNormal(
                loc = self.loc.to_units(new_units, force=force),
                scale = self.scale.to_units(new_units, force=force)
            )
        except DimensionalityError as e:
            print(f"cannot convert {self.units} to {new_units}. if you'd like to assign new units, use force=True", file=sys.stderr)
            raise e

    @property
    def dimensionality(self):
        return self.loc.dimensionality

    @property
    def magnitude(self):
        return self.loc.magnitude

    @property
    def units(self):
        return self.loc.units

    @property
    def internal_units(self):
        return get_internal_units(self.loc)

    def __repr__(self):
        if bellini.verbose:
            return super().__repr__()
        else:
            if not isinstance(self.loc, Distribution):
                u = f'{self.loc:~P.3e}'
            else:
                u = f'{self.loc}'
            if not isinstance(self.scale, Distribution):
                sig = f'{self.scale:.3e~P}'
            else:
                sig = f'{self.scale}'

            return f'LogNorm({u}, {sig})[{self.units}]'


def gen_lognorm(loc, scale):
    assert loc.dimensionality == scale.dimensionality
    #print("enter lognorm", loc, scale)
    loc_units, scale_units = loc.units, scale.units

    if isinstance(loc, (bellini.Quantity, bellini.Distribution)):
        loc = loc.unitless()
    if isinstance(scale, (bellini.Quantity, bellini.Distribution)):
        scale = scale.unitless()

    u = F.log(loc ** 2 / F.sqrt(loc ** 2 + scale ** 2))
    sig = F.log(1 + scale ** 2 / loc ** 2)

    if isinstance(u, bellini.Quantity):
        #print("u is quant")
        if u.dimensionality == ureg.dimensionless.dimensionality:
            u = u.to_units(loc_units, force=True)
    elif isinstance(u, bellini.Distribution):
        #print("u is dist", u)
        u = u.to_units(loc_units, force=True)
    else:
        #print("u is scalar")
        #print("gen_lognorm", u)
        u = bellini.Quantity(u, loc_units)

    if isinstance(sig, bellini.Quantity):
        if sig.dimensionality == ureg.dimensionless.dimensionality:
            sig = sig.to_units(loc_units, force=True)
        #print("sig is quant")
    elif isinstance(sig, bellini.Distribution):
        #print("sig is dist", sig)
        sig = sig.to_units(scale_units, force=True)
    else:
        #print("sig is scalar")
        #print("gen_lognorm", u)
        sig = bellini.Quantity(scale, scale_units)

    #print(u, sig)

    ret = LogNormal(
        loc=u,
        scale=sig
    )
    #print(ret)
    return ret
