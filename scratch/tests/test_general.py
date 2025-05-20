"""
General functionality tests
"""

from pytest import approx


def test_pdf():
    """compare filterpy to sklearn.stats"""
    from scipy.stats import norm
    import filterpy.stats

    a = norm(2, 3).pdf(1.5)
    b = filterpy.stats.gaussian(x=1.5, mean=2, var=3 * 3)
    assert a == approx(b)
