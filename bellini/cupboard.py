# =============================================================================
# IMPORTS
# =============================================================================
import abc
import pint
from .node import Node
from .distributions import Distribution
ureg = pint.UnitRegistry()

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Species(Node):
    def __init__(self, name):
        super(Species, self).__init__()
        self.name = name

    def __eq__(self, other):
        if not isinstance(other, Species):
            return False
        return self.name == other.name

    def __mul__(self, x):
        assert isinstance(x, Distribution)
        return Substance(self, x)

    def __repr__(self):
        return self.name

class Substance(Node):
    def __init__(self, species, x):
        super(Substance, self).__init__()
        assert isinstance(species, Species)
        assert isinstance(x, Distribution)
        assert x.unit == ureg.mole
        self.parents = {"species": species, "x": x}

    def __eq__(self, other):
        if not isinstance(other, Substance):
            return False
        return self.species == other.species and self.x == other.x

    def __repr__(self):
        return "%s * %s" % (self.x, self.species)

class Mixture(Node):
    def __init__(self, substances):
        super(Mixture, self).__init__()
        assert len(
            set([substance.species for substance in substances])
        ) == len(substances)

        for substance in substances:
            assert isinstance(substance, Substance)
            self.children[substance.name] = substance.x

        self.substances = substances

    def __repr__(self):
        return "+".join(self.substances)
