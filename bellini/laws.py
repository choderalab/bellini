import abc
import bellini
from bellini.units import *
from bellini.quantity import Quantity
import numpy as np
import jax.numpy as jnp

class Law(abc.ABC):
    def __init__(self, input_mapping):
        self.input_mapping = input_mapping

    @abc.abstractmethod
    def apply(self, group):
        """ apply should return:
            - a dict of label -> output
            - a map of edges from inputs labels (from the group)
              to output labels (those returned)
            apply should also set all input labels to properties in the group.
            this is just for easier graph gen during _build_graph
            TODO: is it a good idea to induce these input labels?
        """
        raise NotImplementedError()

def log(x):
    if isinstance(x, bellini.distributions.Distribution):
        return x.log()
    elif isinstance(x, bellini.quantity.Quantity):
        return Quantity(jnp.log(x.magnitude), ureg.dimensionless)
    else:
        return Quantity(jnp.log(x), ureg.dimensionless)

def exp(x):
    if isinstance(x, bellini.distributions.Distribution):
        return x.exp()
    elif isinstance(x, bellini.quantity.Quantity):
        return Quantity(jnp.exp(x.magnitude), ureg.dimensionless)
    else:
        return Quantity(jnp.exp(x), ureg.dimensionless)

def sqrt(x):
    if isinstance(x, bellini.distributions.Distribution):
        return x ** 0.5
    else:
        return jnp.sqrt(x)

class TwoComponentBindingModel(Law):
    def __init__(self, input_mapping):
        assert "ligand" in input_mapping.keys()
        assert isinstance(
            input_mapping["ligand"],
            bellini.groups.Species
        )
        assert "protein" in input_mapping.keys()
        assert isinstance(
            input_mapping["protein"],
            bellini.groups.Species
        )
        assert "dG" in input_mapping.keys()
        """
        # could also be Quantity? i feel like we usually want to infer this though
        assert isinstance(
            input_mapping["dG"],
            bellini.distributions.Distribution
        )
        """
        super(TwoComponentBindingModel, self).__init__(input_mapping)

    def apply(self, solution):
        ligand = self.input_mapping["ligand"]
        protein = self.input_mapping["protein"]
        dG = self.input_mapping["dG"]
        Ltot = solution.concentrations[ligand]
        Ptot = solution.concentrations[protein]
        # set properties for graph generation later
        solution.ligand = Ltot
        solution.protein = Ptot
        solution.dG = dG

        MOLAR = ureg.mole / ureg.liter
        Ltot = Ltot.to(MOLAR)
        Ptot = Ptot.to(MOLAR)
        #Ltot, L_units = Ltot.to(MOLAR).magnitude, Ltot.units
        #Ptot, P_units = Ptot.to(MOLAR).magnitude, Ptot.units
        #dG, dG_units = dG.magnitude, dG.units

        """
        # Handle only strictly positive elements---all others are set to zero as constants
        try:
            nonzero_indices = jnp.where(Ltot > 0)[0]
            zero_indices = jnp.where(Ltot <= 0)[0]
        except:
            nonzero_indices = jnp.array(range(Ltot.shape[0]))
            zero_indices = jnp.array([])
        nnonzero = len(nonzero_indices)
        nzeros = len(zero_indices)

        # Numerically stable variant
        dtype = jnp.float32
        Ptot = Ptot.astype(dtype)  # promote to dtype
        Ltot = Ltot.astype(dtype)  # promote to dtype
        PL = jnp.zeros(Ptot.shape, dtype)
        logP = jnp.log(jnp.take(Ptot, nonzero_indices))
        logL = jnp.log(jnp.take(Ltot, nonzero_indices))
        logPLK = jnp.logaddexp(jnp.logaddexp(logP, logL), dG)
        PLK = jnp.exp(logPLK)
        sqrt_arg = 1.0 - jnp.exp(jnp.log(4.0) + logP + logL - 2.0 * logPLK)
        sqrt_arg = jnp.where(sqrt_arg >= 0.0, sqrt_arg, 0)  # ensure always positive
        PL = PL.at[nonzero_indices].set(
            0.5 * PLK * (1.0 - jnp.sqrt(sqrt_arg))
        )  # complex concentration (M)
        """
        dtype = jnp.float32
        Ptot = Ptot.astype(dtype)  # promote to dtype
        Ltot = Ltot.astype(dtype)  # promote to dtype
        PL = jnp.zeros(Ptot.shape, dtype)
        logP = log(Ptot)
        logL = log(Ltot)
        print(logP, logL)
        logPLK = log(exp(logP) + exp(logL) + exp(dG))
        print(logPLK)
        PLK = exp(logPLK)
        print(PLK)
        sqrt_arg = 1.0 - exp(log(4.0) + logP + logL - 2.0 * logPLK)
        print(sqrt_arg)
        PL = 0.5 * PLK * (1.0 - sqrt(sqrt_arg)) # complex concentration (M)
        print(PL)

        # Compute remaining concentrations.
        P = Ptot - PL
        # free protein concentration in sample cell after n injections (M)
        L = Ltot - PL
        # free ligand concentration in sample cell after n injections (M)

        """
        # Ensure all concentrations are within limits, correcting cases where numerical issues cause problems.
        PL = jnp.where(PL >= 0.0, PL, 0.0)  # complex cannot have negative concentration
        P = jnp.where(P >= 0.0, P, 0.0)
        L = jnp.where(L >= 0.0, L, 0.0)
        """

        print(PL.units)
        PL = Quantity(PL, MOLAR).to(P_units)
        P = Quantity(P, MOLAR).to(P_units)
        L = Quantity(L, MOLAR).to(L_units)

        # generate input-output edges
        io_map = []
        for i in ["ligand", "protein", "dG"]:
            for o in ["conc_P", "conc_L", "conc_PL"]:
                io_map.append((i, o))

        return {
            "conc_P": P,
            "conc_L": L,
            "conc_PL": PL
        }, io_map
