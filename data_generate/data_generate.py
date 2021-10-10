import math
import numpy as np
# can get the a*sin(b*(x+c)),and return a tuple list with (x, f(x))
# use the like: g_list[0] = (x0, f(x0))




def normal_noise(x, left=-0.1, right=0.1):
    return x + np.random.rand(len(x))


def uniform_noise(x, left=-0.1, right=0.1):
    return x + np.random.uniform(left, right)


def generate_data(function, noise=uniform_noise,  n=10, left=-1.0, right=1.0):
    result_x = []
    result_y = []
    path = (right - left)/(n - 1)
    x = left
    for i in range(n):
        result_x.append(x)
        result_y.append(noise(function(x)))
        x += path
    return result_x, result_y






