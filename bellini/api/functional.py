import bellini
import pint
import numpy as np
import sys
import warnings
from pint.errors import DimensionalityError

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
    def wrapped_fn(*args, **kwargs):
        is_dist = np.array([
            isinstance(arg, bellini.Distribution)
            for arg in args
        ])
        is_quantity = np.array([
            isinstance(arg, bellini.Quantity)
            for arg in args
        ])
        """
        is_unitless = np.array([
            not hasattr(arg, "units")
            for arg in args
        ])
        """
        is_unitless = np.array(~(is_dist | is_quantity))

        """
        #print(is_dist, is_quantity, is_unitless, fn)
        assert (is_dist | is_quantity | is_unitless).all(), (
            "bellini.api.functional only takes Quantities"
            ", Distributions, and unitless scalars/arrays as args"
        )
        """

        if is_unitless.all():
            #print(fn, type(fn))
            return fn(*args, **kwargs)
        else:
            if is_unitless.any():
                warnings.warn(("all arguments with no units are being set to dimensionless Quantities."
                             " if this is not the desired behavior, consider stripping"
                             " units first and adding them on afterwards."))
                args = [
                    arg if isinstance(arg, (bellini.Quantity, bellini.Distribution)) else bellini.Quantity(arg, ureg.dimensionless)
                    for arg in args
                ]

            #print("args", args, fn)

            is_quantity = np.array([
                isinstance(arg, bellini.Quantity)
                for arg in args
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

    return wrapped_fn

def __getattr__(name):
    if name in OPS.keys():
        return OPS[name]
    else:
        if bellini.infer:
            import jax.numpy as jnp
            return _fn_wrapper(getattr(jnp, name))
        else:
            return _fn_wrapper(getattr(np, name))
