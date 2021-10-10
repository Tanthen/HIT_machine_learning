import numpy as np
from model import polyfit
if __name__ == '__main__':
    G = [[3,-1],
         [-1,1]]
    B = [[-2],
         [0]]

    w = [[0],
         [0]]
    g = np.matmul(G, w) + B
    d = -g
    print(d)
    while np.linalg.norm(g) >= 0.00001:
        a = (np.vdot(g, g)) / (np.vdot(d, np.matmul(G, d)))
        print("a=" + str(a))
        w = w + a * d
        g_new = np.matmul(G, w) + B
        beta = np.vdot(g_new, g_new) / np.vdot(g, g)
        print("beta=" + str(beta))

        g = g_new
        print(g)
        d = -g + beta * d
        print(d)

        print(np.linalg.norm(g))



