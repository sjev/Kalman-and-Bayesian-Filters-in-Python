"""
General functionality tests
"""

import scipy.stats
import filterpy.stats
import numpy as np
from pytest import approx


def test_pdf():
    """compare filterpy to sklearn.stats"""

    # NOTE: this works with scalars, but not with arrays

    a = scipy.stats.norm(2, 3).pdf(1.5)
    b = filterpy.stats.gaussian(x=1.5, mean=2, var=3 * 3)
    assert a == approx(b)
