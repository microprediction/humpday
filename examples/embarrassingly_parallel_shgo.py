from shgo import shgo
import time
import numpy as np
from embarrassingly.parallel import Parallel

# Illustrates use of the new SHGO worker pattern in combination with the embarrassingly library
# The idea is that you write a "pre-objective" function which is an objective
# function with one additional prepended argument.

# If you are just going to execute the objective function on a local machine or on a cluster on which
# you have ray multiprocessing set up, you don't need this pattern. You can simply supply f to the
# optimizer.

# However, if f is to be executed somewhere else (e.g. you are shelling out, ssh'ing somewhere etc), or
# you don't have or want to use Ray, you can consider the pattern provided by embarrassingly.parallel.Parallel
# The line F = Parallel(_F, num_workers=NUM_WORKERS) converts the "pre-objective" function _F(i, <f args>)
# into a callable class F(<same args as f>) that can be supplied to an optimization library (such as SHGO in this example).
# However F() is smart enough to maintain a queue and decide which worker should be indicated when _F( ) is called.



NUM_WORKERS = 8


# Suppose the objective function is
def f(x):
    return x[0] ** 2 + x[1] ** 2


# Then we write a "pre-objective function" taking one extra argument, the worker number
def _F(i, x):
    """ Illustrates how to write a pre-objective function """
    # Use the worker number to allocate job somewhere, for example
    time.sleep(0.1)
    print('Sending job to server number '+str(i))
    return f(x)


# And then use Parallel to create F, which expects only x
# F will be smart enough to call _F with a sensible worker number
F = Parallel(_F, num_workers=NUM_WORKERS)


def demo():
    bounds = np.array([[0, 1], ] * 2)

    # Single server ...
    ts = time.time()
    res = shgo(f, bounds, n=50, iters=2)
    print(f'Total time serial: {time.time() - ts}')
    print('-')
    print(f'res = {res}')
    ts = time.time()


    # Multiple servers
    res = shgo(F, bounds, n=50, iters=2, workers=NUM_WORKERS)
    print('=')
    print(f'Total time par: {time.time() - ts}')
    print('-')
    print(f'res = {res}')


if __name__ == '__main__':
    demo()