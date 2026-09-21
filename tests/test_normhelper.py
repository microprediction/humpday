import pytest

# numpy is the `fast` extra rather than a dependency, and CI runs the whole suite on the
# dependency-free backend too. A test that needs numpy to express itself skips there rather
# than failing to import, which is how this suite already treats its optional dependencies.
np = pytest.importorskip("numpy")

from humpday.transforms.normhelper import NormHelper


def test_cdf_invcdf():
    normcdf = NormHelper._normcdf_function()
    norminv = NormHelper._norminv_function()
    for x in np.random.randn(100):
        x1 = norminv(normcdf(x))
        assert abs(x - x1) < 1e-4


if __name__ == "__main__":
    test_cdf_invcdf()
