from humpday.transforms.normhelper import NormHelper
import numpy as np


def test_cdf_invcdf():
    normcdf = NormHelper._normcdf_function()
    norminv = NormHelper._norminv_function()
    for x in np.random.randn(100):
        x1 = norminv(normcdf(x))
        assert abs(x-x1)<1e-4


if __name__=='__main__':
    test_cdf_invcdf()