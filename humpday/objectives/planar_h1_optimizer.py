import numpy as np
from timemachines.common.plotting import mesh2d
from timemachines.objectives.planar_h1 import hump_selector
from shgo import shgo
from sklearn.datasets import make_spd_matrix

# Just messin' around here.
# Moves h1-like hump functions around on the plane until
# finding the minima of them because easier.


def locations_objective(zs, random_ws, offset):
    """ Return mean minimum found
        zs - center of humps
        ws - weights
        offset - offset to apply to hump function
    """
    scores = list()


    for ws_ in random_ws:
        def hs(xs):
            return hump_selector(xs=xs,zs=zs,ws=ws_,offset=offset)
        dim = 2
        bounds = [(-0.5,0.5)]*dim
        res = shgo(hs, bounds, n=10, iters=2)
        score = 100*res.fun + res.nfev
        scores.append(score)
    return np.mean(scores)


def best_locations(ws,offset, random_ws):
    """
        For given weights on humps, move them around until it helps the search

        z0   Initial locations of humps in flat format [ x1, y1, x2, y2, ... ]
        ws   Weights assigned to each hump
    """

    # Simulate random hump multipliers


    def mb(zs):
        return locations_objective(zs=np.array(zs), random_ws=random_ws, offset=offset)

    dim = int(len(ws))
    bounds = [(-0.5,0.5)]*dim*2
    res = shgo(mb, bounds, n=100, iters=50)
    print(res.fun)
    return res.x


if __name__=='__main__':

    ws = np.array([0.5,0.5,0.5])

    dim = len(ws)
    zs = 0.333*np.random.rand(2*dim)
    offset = 0.1 * np.random.randn(2)
    zdim = len(ws)
    num_ws = 10

    C = make_spd_matrix(zdim)
    while (C[0][1])**2/(C[0][0]*C[1][1])<0.97:
        C = make_spd_matrix(zdim)
    print(C)
    random_ws = np.random.multivariate_normal(ws, 2*C, num_ws)

    avg = locations_objective(zs=zs, random_ws=random_ws, offset=offset)
    print(avg)

    # Optimize positions of humps making it easier
    zBest = best_locations(ws=ws,offset=offset,random_ws=random_ws)
    print(zBest)
    print(ws)
    import random
    f = lambda x1, x2: hump_selector(xs=np.array([x1, x2]), zs=zBest, ws=random.choice(random_ws), offset=offset)
    mesh2d(f)
    pass



