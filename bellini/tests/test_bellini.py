"""
Unit and regression test for the bellini package.
"""

# Import package, test suite, and other packages as needed
import bellini
import pytest
import sys

def test_bellini_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "bellini" in sys.modules
