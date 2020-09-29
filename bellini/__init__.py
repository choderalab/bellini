"""
bellini
Bayesian Learning on Laboratory Investigations
"""

# Add imports here
from . import quantity, groups, distributions

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
