try:
    from humpday.transforms.cubetosimplex import minimize_optimizer_on_simplex
    using_winning = True
except ImportError:
    using_winning = False
