"""
bellini
Bayesian Learning on Laboratory Investigations
"""

# Add imports here
from ._globals import verbose
from .quantity import *
from .groups import *
from .distributions import *
from .story import *
from . import quantity, groups, distributions, story


# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
