from shgo import shgo
import time
import numpy as np


# Toy problem
def f(x):
    time.sleep(0.1)
    return x[0] ** 2 + x[1] ** 2


def demo():
    bounds = np.array([[0, 1], ] * 2)

    ts = time.time()
    res = shgo(f, bounds, n=50, iters=2)
    print(f'Total time serial: {time.time() - ts}')
    print('-')
    print(f'res = {res}')
    ts = time.time()
    res = shgo(f, bounds, n=50, iters=2, workers=8)
    print('=')
    print(f'Total time par: {time.time() - ts}')
    print('-')
    print(f'res = {res}')


if __name__ == '__main__':
    demo()