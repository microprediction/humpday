from microconventions.zcurve_conventions import ZCurveConventions
from typing import List
import numpy as np
import math

# A mostly failed experiment, thus far.

###########################################################
#                                                         #
# Part I: Constructing an embedding   [0,1]^m -> [0,1]^n  #
#                                                         #
###########################################################

def to_space(p: float, bounds=None, dim: int = None):
    """ Interprets p as a point in a rectangle in R^2 or R^3
         :param bounds  [ (low,high), (low,high), (low,high) ] defaults to unit cube
         :param dim     Dimension. Only used if bounds are not supplied.
    """
    if bounds is None:
        assert dim is not None, "If you don't supply bounds, dimension of hypercube is required"
        bounds = [(0, 1) for _ in range(dim)]
    else:
        dim = len(bounds)

    if dim > 1:
        us = reversed(ZCurveConventions().to_cube(zpercentile=p, dim=dim))  # 0 < us[i] < 1
    else:
        us = [p]
    return [u * (b[1] - b[0]) + b[0] for u, b in zip(us, bounds)]


def from_space(ps: [float], bounds=None) -> float:
    """ [ , ]^n -> [0,1] """
    if bounds is None:
        bounds = [(0, 1) for _ in range(len(ps))]
    us = [(pi - b[0]) / (b[1] - b[0]) for pi, b in zip(ps, bounds)]
    for u in us:
        assert 0 <= u <= 1, "bounds are inconsistent with p=" + str(ps)
    if len(us) > 1:
        return ZCurveConventions().from_cube(list(reversed(us)))
    else:
        return us[0]


def chunk_to_end(l: List, n: int) -> List[List]:
    """ Break list in to evenly sized chunks
        :param n: Size of batches
    """
    rl = list(reversed(l))
    chunks = [list(reversed(rl[x:x + n])) for x in range(0, len(rl), n)]
    return list(reversed(chunks))


def flatten(l: List[List]) -> List:
    return [item for sublist in l for item in sublist]


def curl(u:[float], d:int)->[float]:
    """ Cuts down the dimension by about 1/d
          [0,1]^m -> [0,1]^n
          d :  How many dimensions to fold into 1
    """
    assert d in [2, 3]
    u_chunks = chunk_to_end(u, d)
    return [from_space(uc) for uc in u_chunks]


def uncurl(v:[float], d:int, n_dim:int)->[float]:
    """ Inverse of curl, approximately
         n_dim:
    """
    assert d in [2, 3]
    # First infer the dimension to unfold the first entry into
    u_chunks = chunk_to_end([1 for _ in range(n_dim)], d)
    n_last = len(u_chunks[0])
    if n_last<d:
        u0_dim = len(u_chunks[0])
        # Unfold, u[0] is a special case
        u0 = to_space(v[0], dim=u0_dim)
        u_rest = [to_space(vj, dim=d) for vj in v[1:]]
        return u0 + flatten(u_rest)
    else:
        return flatten([to_space(vj, dim=d) for vj in v])


#######################################################################
#                                                                     #
# Part II: Running optimizers in a lower dimension than the problem   #
#                                                                     #
#######################################################################


def verify_embedding(embedding, inverse, n_dim):
    """ Get the smaller dimension, but also run some checks
    :param embedding: [0,1]^n -> [0,1]^m
    :param inverse:   [0,1]^m -> [0,1]^n
    :return:
    """
    for _ in range(500):
        u0 = list(np.random.rand(n_dim))
        v0 = embedding(u0)
        assert all([0 <= v0_j <= 1 for v0_j in v0]), "pi does not go to the cube"
        u0_check = inverse(v0, n_dim)
        assert np.linalg.norm(np.array(u0_check) - np.array(u0)) < 1e-1, 'map_down does not invert map_up'
    return len(v0)


def embedding_optimizer_factory(optimizer, objective, n_trials, n_dim, with_count, embedding, inverse):
    """ Apply dlib in a lower dimension
           objective: [0,1]^n -> float
           map_down: A bijection from [0,1]^n -> [0,1]^m
           map_up:   The inverse of map_down
    """
    m_dim = verify_embedding(embedding=embedding, inverse=inverse, n_dim=n_dim)


    def _objective(v) -> float:
        """  [0,1]^m -> """
        u = inverse(v, n_dim)
        return objective(u)

    best_val, best_v, feval_count = optimizer(objective=_objective, n_trials=n_trials, n_dim=m_dim, with_count=True)
    best_x = inverse(best_v, n_dim)

    return (best_val, best_x, feval_count) if with_count else (best_val, best_x)


def curl_factory(optimizer, objective, n_trials, n_dim, with_count, d):
    assert n_trials>3*(n_dim+1)
    ranking = importance(objective,n_dim)  # Wastes n_dim+1 trials at least, so pretty stupid
                                           # Just a quick experiment
    ordering = grouped_ordering(ranking,d)
    try:
        inv_ordering = [ ordering.index(j) for j in range(n_dim) ]
    except ValueError:
        print('groan')
        pass

    def permute(u):
        return [ u[j] for j in ordering ]

    def permute_inv(u):
        try:
            return [ u[j] for j in inv_ordering ]
        except IndexError:
            print('what the')
            pass

    debug = True # Remove this crud at some point
    if debug:
        u0 = list(np.random.rand(n_dim))
        u0_check = permute_inv(permute(u0))
        assert np.linalg.norm(np.array(u0)-np.array(u0_check))<1e-6

    def curld(u:[float])->[float]:
        u_permute = permute(u)
        return curl(u_permute,d)

    def uncurld(v:[float], n_dim:int)->[float]:
        u = uncurl(v=v,d=d,n_dim=n_dim)
        return permute_inv(u)

    return embedding_optimizer_factory(optimizer=optimizer, objective=objective, n_trials=n_trials-2*n_dim-2, n_dim=n_dim,
                                       with_count=with_count, embedding=curld, inverse=uncurld)


def dlib_curl_cube3(objective, n_trials, n_dim, with_count):
    """ Curled up version of dlib optimizer """
    return curl_factory(optimizer=dlib_cube, objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, d=3)


#######################################################################
#                                                                     #
# Part III: Dreadful hacks for coordinate importance                  #
#                                                                     #
#######################################################################
from scipy.stats import rankdata


def importance(objective, n_dim):
    # Wasteful and crude
    u1 = [0.3*(x-0.5)+0.5 for x in np.random.rand(n_dim) ]
    u2 = [0.3 * (x - 0.5) + 0.5 for x in np.random.rand(n_dim)]
    f1 = objective(u1)
    f2 = objective(u2)
    abs_derivs = list()
    for j in range(n_dim):
       uj1 = u1
       uj1[j] = uj1[j]+0.1
       fj1 = objective(uj1)
       d1 = (fj1-f1)/0.1
       uj2 = u2
       uj2[j] = uj2[j] + 0.01
       fj2 = objective(uj2)
       d2 = (fj2 - f2)/0.01
       d = 0.5*d1 + 0.5*d2
       abs_derivs.append(abs(d))

    ordering = [ j for d,j in sorted(list(zip( abs_derivs, range(n_dim))),reverse=True) ]
    return ordering


def grouped_ordering(ordering,d):
    """ A permutation of coordinates that might work with curl( ,d)
        e.g. if d=2 this alternates between important and unimportant variables
        The most important variables might not be curled at all.
    """
    n_dim = len(ordering)
    n_group = int(math.floor(n_dim/d))
    n_rem = n_dim-n_group*d

    the_head = ordering[:n_rem]
    the_tail = ordering[n_rem:]

    tail_groups = [ the_tail[j:j + n_group] for j in range(0, len(the_tail), n_group)]
    tail_ordering = flatten(list(map(list, zip(*tail_groups))))
    return the_head + tail_ordering








if __name__ == '__main__':
    from humpday.objectives.classic import deap_combo1_on_cube as an_objective
    from humpday.optimizers.dlibcube import dlib_cube

    best_val, best_x, feval_count = dlib_cube(objective=an_objective, n_trials=50, n_dim=14, with_count=True)
    best_val_0, best_x_0, feval_count_0 = dlib_curl_cube3(objective=an_objective, n_trials=150, n_dim=14,
                                                          with_count=True)
    pass
    # The second version works in higher dimensions
    best_val_2, best_x_2, feval_count_2 = dlib_curl_cube3(objective=an_objective, n_trials=150, n_dim=34,
                                                          with_count=True)
