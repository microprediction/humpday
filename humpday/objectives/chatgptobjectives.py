import numpy as np


def chat_0(u):
    n_dim = len(u)
    u = np.array(u)
    s = np.sin(np.pi * u)
    q = np.sum((np.outer(u, u) - np.eye(n_dim)) ** 2)
    return np.sum(s * q)


import numpy as np


# Chat Function 1
def chat_1(u):
    """
    A difficult optimization problem that involves finding the minimum of a
    function with a narrow valley and many local minima. The function is
    defined in terms of the sum of two cosines and the absolute difference
    between adjacent coordinates of the input vector.
    """
    n_dim = len(u)
    term1 = np.sum(np.cos(u))
    term2 = np.sum(np.abs(np.diff(u)))
    return -term1 - term2 + n_dim


# Chat Function 2
def chat_2(u):
    """
    An optimization problem that involves minimizing the sum of the squared
    differences between each coordinate of the input vector and a scaled
    version of the same coordinate in the previous step. This function has
    many local minima and can be difficult to optimize.
    """
    n_dim = len(u)
    u = np.array(u)
    term1 = np.sum(u ** 2)
    term2 = np.sum(((u - 0.1 * np.roll(u, 1)) ** 2)[:-1])
    return term1 + term2


# Chat Function 3
def chat_3(u):
    """
    An optimization problem that involves finding the minimum of a function
    with a narrow valley and many local minima. The function is defined in
    terms of the sum of a scaled and shifted sine function and the sum of
    the absolute differences between adjacent coordinates of the input
    vector raised to a power.
    """
    n_dim = len(u)
    term1 = 0.1 * np.sum(np.sin(3 * u))
    term2 = np.sum(np.abs(np.diff(u)) ** 0.5)
    return term1 + term2


# Chat Function 4
def chat_4(u):
    """
    An optimization problem that involves finding the minimum of a function
    with a narrow valley and many local minima. The function is defined in
    terms of the sum of a scaled and shifted cosine function and the product
    of adjacent coordinates of the input vector raised to a power.
    """
    n_dim = len(u)
    term1 = 0.1 * np.sum(np.cos(3 * u))
    term2 = np.prod(np.abs(np.diff(u)) ** 1.5)
    return term1 - term2


# Chat Function 5
def chat_5(u):
    """
    An optimization problem that involves minimizing a modified version of
    the Rosenbrock function. The modification involves adding a scaled
    version of the sum of the absolute differences between adjacent
    coordinates of the input vector.
    """
    n_dim = len(u)
    u = np.array(u)
    y = 0
    for i in range(n_dim - 1):
        term1 = (u[i + 1] - u[i] ** 2) ** 2
        term2 = (1 - u[i]) ** 2
        y += 100 * term1 + term2
    y += 0.1 * np.sum(np.abs(np.diff(u)))
    return y


# Chat Function 6
def chat_6(u):
    """
    An optimization problem that involves minimizing a modified version of
    the Griewank function. The modification involves adding a scaled version
    of the sum of the squared differences between adjacent coordinates of
    the input vector.
    """
    n_dim = len(u)
    u = np.array(u)
    term1 = np.sum(u ** 2) / 4000
    term2 = np.prod(np.cos(u / np.sqrt(np.arange(1, n_dim + 1))))
    term3 = 0
    for i in range(n_dim - 1):
        term3 += (u[i + 1] - u[i]) ** 2
    term3 *= 0.1
    return term1 - term2 + term3 + 1


# Chat Function 7
def chat_7(u):
    """
    An optimization problem that involves finding the minimum of a function
    with a narrow valley and many local minima. The function is defined in
    terms of the sum of a scaled and shifted sine function and the product
    of adjacent coordinates of the input vector raised to a power.
    """
    n_dim = len(u)
    u = np.array(u)
    term1 = 0.1 * np.sum(np.sin(3 * u))
    term2 = np.prod(np.abs(np.diff(u)) ** 2)
    return term1 - term2


def chat_8(u):
    """
    A modified version of the original chat_0 function that includes a
    non-linear transformation of the input vector. This transformation is
    designed to introduce additional local minima and make the optimization
    problem more challenging.
    """
    n_dim = len(u)
    u = np.array(u)
    u = np.sin(u) * np.exp(u)
    q = np.sum((np.outer(u, u) - np.eye(n_dim)) ** 2)
    return np.sum(u * q)


# Sum of Squares Function
def chat_9(u):
    u = np.array(u)
    return np.sum(u ** 2) + 0.1*np.sum((u-0.1)**4)


# Egg Holder Function
def chat_10(u):
    """
    An optimization problem that involves finding the minimum of a function
    that has a complex egg-carton shape. This function has many local minima
    and can be challenging to optimize.
    """
    u = np.array(u)
    term1 = -(u[1] + 47) * np.sin(np.sqrt(np.abs(u[1] + u[0] / 2 + 47)))
    term2 = -u[0] * np.sin(np.sqrt(np.abs(u[0] - (u[1] + 47))))
    return term1 + term2


# Six-Hump Camel Function
def chat_11(u):
    """
    An optimization problem that involves finding the minimum of a function
    with six local minima, two of which are global. The function has a
    "camel-back" shape and can be challenging to optimize.
    """
    u = np.array(u)
    term1 = (4 - 2.1 * u[0] ** 2 + (u[0] ** 4) / 3) * u[0] ** 2
    term2 = u[0] * u[1]
    term3 = (-4 + 4 * u[1] ** 2) * u[1] ** 2
    return term1 + term2 + term3


# Branin Function
def chat_12(u):
    """
    An optimization problem that involves finding the global minimum of a
    function with three local minima. The function has a combination of
    oscillations and a narrow valley, which makes it chat to optimize.
    """
    u = np.array(u)
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    term1 = a * (u[1] - b * u[0] ** 2 + c * u[0] - r) ** 2
    term2 = s * (1 - t) * np.cos(u[0]) + s
    return term1 + term2


import numpy as np


# Modified Sphere Function
def chat_13(u):
    """
    An optimization problem that involves minimizing a modified version of 
    the classic Sphere function. The modification involves multiplying each 
    coordinate of the input vector by a function of its index, which adds 
    nonlinearity and complexity to the landscape.
    """
    n_dim = len(u)
    u = np.array(u)
    f = np.arange(1, n_dim + 1)
    return np.sum(f * u ** 2)


# Griewank Function with Sinusoidal Perturbation
def chat_14(u):
    """
    An optimization problem that involves minimizing the Griewank function 
    with a sinusoidal perturbation added to one coordinate of the input 
    vector. The perturbation adds complexity to the landscape and can make 
    it more chat to optimize.
    """
    n_dim = len(u)
    u = np.array(u)
    term1 = np.sum(u ** 2) / 4000
    term2 = np.prod(np.cos(u / np.sqrt(np.arange(1, n_dim + 1))))
    perturb = np.sin(10 * u[0])
    return term1 - term2 + perturb


# Modified Rastrigin Function
def chat_15(u):
    """
    A modified version of the Rastrigin function that introduces a 
    non-linear transformation of the input vector. This transformation 
    involves taking the absolute value of each coordinate and then 
    raising it to a power that increases with the index. This adds 
    complexity and nonlinearity to the landscape.
    """
    n_dim = len(u)
    f = np.arange(1, n_dim + 1)
    v = np.abs(u) ** f
    return np.sum(v + 10 * (n_dim - np.sum(np.cos(2 * np.pi * v))))


# Zakharov Function with Sine Perturbation
def chat_16(u):
    """
    An optimization problem that involves minimizing the Zakharov function 
    with a sine perturbation added to one coordinate of the input vector. 
    The perturbation adds complexity to the landscape and can make it more 
    chat to optimize.
    """
    n_dim = len(u)
    u = np.array(u)
    term1 = np.sum(u ** 2) + np.sum(0.5 * np.arange(1, n_dim + 1) * u) ** 2 + \
            np.sum(0.5 * np.arange(1, n_dim + 1) * u) ** 4
    perturb = np.sin(10 * u[0])
    return term1 + perturb


# Modified Ackley Function
def chat_17(u):
    """
    A modified version of the Ackley function that introduces a non-linear 
    transformation of the input vector. This transformation involves taking 
    the square of each coordinate and then adding it to the product of 
    adjacent coordinates. This adds complexity and nonlinearity to the 
    landscape.
    """
    n_dim = len(u)
    u = np.array(u)
    v = u ** 2 + np.roll(u, -1) * np.roll(u, 1)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(v) / n_dim))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * v)) / n_dim)
    return term1 + term2 + 20 + np.exp(1)


CHATGPT_OBJECTIVES = [chat_0, chat_1, chat_2, chat_3, chat_4,chat_5, chat_6, chat_7,
                     chat_8, chat_9, chat_10, chat_11, chat_12, chat_13, chat_14,
                     chat_15, chat_16, chat_17]

if __name__=='__main__':
    for obj in CHATGPT_OBJECTIVES:
        for u in [ [1,2], [1.0,0,0,0,0,0]]:
            obj(u)