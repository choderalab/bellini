"""
bellini
Bayesian Learning on Laboratory Investigations
"""

# Add imports here
from .quantity import *
from .groups import *
from .distributions import *
from .procedure import *
from . import quantity, groups, distributions, procedure

def set_verbose(v = True):
    global verbose
    verbose = v
set_verbose(False)

_backend_default="numpyro"
_supported_backends = ["numpyro"]
def set_backend(bk):
    global backend
    if bk in _supported_backends:
        backend = bk
    else:
        raise ValueError(f"backend {backend} is not supported")
set_backend(_backend_default)

class inference:
    def __init__(self, _infer = True):
        global infer
        self._old_infer = infer
        self._new_infer = _infer
    def __enter__(self):
        global infer
        self._old_infer = infer
        infer = self._new_infer
    def __exit__(self, type, value, traceback):
        global infer
        infer = self._old_infer


def set_infer(i = False):
    global infer
    infer = i
set_infer(False)

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
