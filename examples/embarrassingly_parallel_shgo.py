from shgo import shgo
import time
import numpy as np
from embarrassingly.parallel import Parallel

# Illustrates use of the new SHGO worker pattern in combination with the embarrassingly library
# The idea is that you write a "pre-objective" function which is an objective
# function with one additional prepended argument.

NUM_WORKERS = 8


# Suppose the objective function is
def f(x):
    return x[0] ** 2 + x[1] ** 2


# Then we write a "pre-objective function" taking one extra argument, the worker number
def _F(i, x):
    """ Illustrates how to write a pre-objective function """
    # Use the worker number to allocate job somewhere
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