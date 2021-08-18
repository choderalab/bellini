""" A high-level interface that provides a backend-agnostic numpy-like API when
performing computations on Quantities and Distributions

bellini.api.functional serves as a wrapper for backend tensor-accelerated framework
operations. For the most part it doesn't contain functions itself, and instead
wraps whatever function you query from it in a way that you can apply it to
things like Quantities or Distributions. Hence, you should import the module
instead of attempting to import functions directly from it. A convenient
nomenclature, inspired by pytorch, is :code:`import bellini.api.functional as F`.

For example, say we want to take a log of a Normal Distribution. We can do this
using the functional API as such

.. highlight:: python
.. code-block:: python

    from bellini.quantity import Quantity as Q
    from bellini.distributions import Normal
    import bellini.api.functional as F

    pre_log_normal = Normal(Q(0), Q(1)) # unitless distribution
    log_normal = F.log(pre_log_normal)
    # type(log_normal) = TransformedDistribution
"""

import bellini
import pint
import numpy as np
from jax import lax
from numpyro.contrib.control_flow import scan
import sys
import warnings
from pint.errors import DimensionalityError
from bellini.units import ureg
from bellini.api.utils import flatten, _to_x_constructor

# =============================================================================
# CONSTANTS
# =============================================================================
OPS = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y,
    "neg": lambda x: -x,
    "pow": lambda x, y: x ** y,
    "slice": lambda x, y: x[y],
}

def _fn_wrapper(fn):
    def _wrapped_fn(*args, **kwargs):

        flat_args = flatten(args)
        is_dist = np.array([
            isinstance(arg, bellini.Distribution)
            for arg in flat_args
        ])
        is_quantity = np.array([
            isinstance(arg, bellini.Quantity)
            for arg in flat_args
        ])
        is_unitless = np.array(~(is_dist | is_quantity))
        #print(is_dist, is_quantity, is_unitless)

        if is_unitless.all():
            ret = fn(*args, **kwargs)
            if isinstance(ret, pint.Quantity):
                ret = bellini.Quantity(ret.magnitude, ret.units)
            return ret
        else:
            if is_unitless.any():
                warnings.warn(("all arguments with no units are being set to dimensionless Quantities."
                             " if this is not the desired behavior, consider stripping"
                             " units first and adding them on afterwards."))

                def _to_dimless_quantity(arg):
                    if isinstance(arg, (bellini.Quantity, bellini.Distribution)):
                        return arg
                    else:
                        return bellini.Quantity(arg, ureg.dimensionless)

                to_dimless_quantity = _to_x_constructor(_to_dimless_quantity)

                args = to_dimless_quantity(args)

            is_quantity = np.array([
                isinstance(arg, bellini.Quantity)
                for arg in flat_args
            ])

            if is_dist.any():
                return bellini.TransformedDistribution(args, op=fn.__name__, **kwargs)
            else:
                assert is_quantity.all(), "@alexjli you didn't account for ur conditionals properly"
                try:
                    ret = fn(*args,**kwargs)
                    # it seems like if you apply an np function to our custom Quantity class
                    # we get the superclass pint.Quantity back :/
                    # so ig we just reconvert it back
                    if isinstance(ret, pint.Quantity):
                        ret = bellini.Quantity(ret.magnitude, ret.units)
                    return ret
                except DimensionalityError as e:
                    print(f"fn {fn} not compatible with args {args} kwargs {kwargs}", file=sys.stderr)
                    print("Consider stripping units before input and attaching them after computation", file=sys.stderr)
                    raise e

    return _wrapped_fn


def functional_for(f, init, xs, length=None):
    fn_to_wrap = None
    if bellini.backend == "numpyro":
        def scan_like(f, init, xs, length):
            init = init.magnitude
            xs = xs.magnitude
            with bellini.inference():
                carry, out = scan(f, init, xs, length)
            return {
                "carry": bellini.Quantity(carry),
                "out": bellini.Quantity(out)
            }

        fn_to_wrap = scan_like

    else:  # this is not tested
        def scan_like(f, init, xs, length):
            if xs is None:
                xs = [None] * length
            carry = init
            ys = []
            for x in xs:
                carry, y = f(carry, x)
                ys.append(y)
            if bellini.backend == "pyro":
                out = torch.stack(ys)
            else:
                out = np.stack(ys)
            return {
                "carry": carry,
                "out": out
            }
        fn_to_warp = scan_like

    sample_out = fn_to_wrap(f, init, xs, length)

    return [
        bellini.distributions._JITDistribution(
            fn_to_wrap,
            {"f": f,
             "init": init,
             "xs": xs,
             "length":length},
             label,
             deterministic_outputs=sample_out
         ) for label in ['carry', 'out']
    ]


def __getattr__(name):
    # TODO: fix this hacky fix that lets both autodoc and pytest to work
    if name == "__path__":
        return [globals()["__file__"]]
    elif name == "__all__":
        print(globals())
        return ["functional_for"]
    elif name in OPS.keys():
        return OPS[name]
    elif name in globals():
        return globals()[name]
    else:
        if bellini.infer:
            import jax.numpy as jnp
            return _fn_wrapper(getattr(jnp, name))
        else:
            return _fn_wrapper(getattr(np, name))
