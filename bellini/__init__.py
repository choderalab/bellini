"""
bellini
Bayesian Learning on Laboratory Investigations
"""

# Add imports here
from .quantity import *
from .groups import *
from .distributions import *
from .story import *
from . import quantity, groups, distributions, story

global verbose
verbose = False

_backend_default="numpyro"
_supported_backends = ["numpyro"]
def set_backend(bk):
    global backend
    if bk in _supported_backends:
        backend = bk
    else:
        raise ValueError(f"backend {backend} is not supported")
set_backend(_backend_default)

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
