import numpy as np
from model import polyfit
from data import *
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    x1 = np.linspace(2, 4.5)
    x2 = np.linspace(-2, 5)
    y1 = 8 * x1 -20
    y2 = x2 ** 2 - 4

    plt.figure(num=3, figsize=(8, 5))
    l1 = plt.plot(x2, y2)
    l2 = plt.plot(x1, y1, color='red', linewidth=1.0, linestyle='--')

    plt.legend(handles=[l1, l2], labels=['up', 'down'], loc='best')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim((-3, 6))
    plt.ylim((-6, 25))

    new_ticks = np.linspace(-2, 6, 9)
    plt.xticks(new_ticks)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))

    plt.show()


