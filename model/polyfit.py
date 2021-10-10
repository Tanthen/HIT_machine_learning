import numpy as np
import math


def get_high_order(x, order):
    result = []
    for i in range(order):
        result.append(x ** i)
    return result


class Poly_Fit:
    def __init__(self):
        super().__init__()

    def initial(self, n, initial_method):
        if initial_method == "uniform":
            return np.random.uniform(size=n)
        elif initial_method == "normal":
            return np.random.rand(n)

    def train(self, X):
        pass

    def parse_result(self, X, Y, order, reg_lambda=0):
        assert len(X) == len(Y)
        n = len(X)
        order = order + 1
        A = []
        for j in range(order):
            row = []
            for i in range(order):
                sum = 0
                for k in range(n):
                    sum += X[k] ** (i + j)
                if i == j:
                    sum += reg_lambda
                row.append(sum)
            A.append(row)
        print(A)

        B = []
        for j in range(order):
            sum = 0
            for k in range(n):
                sum += Y[k] * (X[k] ** j)
            B.append(sum)
        loss = 0
        w = np.linalg.solve(A, B)
        for j in range(n):
            l = 0
            for k in range(order):
                l += w[k] * (X[j] ** k)
            loss += (l - Y[j]) ** 2
        return w, loss

    def grad_result(self, X, Y, order, epoch=1, lr=0.03, reg_lambda=0):
        test_x = []
        it = -1.0
        for i in range(21):
            test_x.append(it + 0.1 * i)
        assert len(X) == len(Y)
        m = len(X)
        w = np.random.rand(order + 1)
        # w = np.array([2,4,6,8,10,12,14,16,18, 20])
        t = []
        y1 = []
        y2 = []
        for e in range(epoch):
            L = []
            for i in range(order + 1):
                sum = 0
                for j in range(m):
                    mx = 0
                    for k in range(order + 1):
                        mx += w[k] * (X[j] ** k)
                    sum += (mx - Y[j]) * (X[j] ** i)
                sum = (sum + reg_lambda * w[i]) * lr / m
                L.append(sum)
            w = w - L
            if e < 1000 or e % 1000 == 0:
                t.append(e)
                loss = 0
                for j in range(m):
                    l = 0
                    for k in range(order + 1):
                        l += w[k] * (X[j] ** k)
                    loss += (l - Y[j]) ** 2
                y1.append(math.sqrt(loss / m))
                loss = 0
                for j in range(21):
                    l = 0
                    for k in range(order + 1):
                        l += w[k] * (test_x[j] ** k)
                    loss += (l - math.sin(2 * math.pi * test_x[j])) ** 2
                y2.append(math.sqrt(loss / 21))

        # loss = 0
        # for j in range(m):
        #     l = 0
        #     for k in range(order+1):
        #         l += w[k]*(X[j]**k)
        #     loss += (l - Y[j])**2

        return w, t, y1, y2

    def conj_grad(self, X, Y, order, reg_lambda=0, threshold=10e-12):
        G = np.zeros((order, order))
        B = np.zeros((order, 1))
        test_x = []
        it = -1.0
        for i in range(21):
            test_x.append(it + 0.1 * i)
        for j in range(len(X)):
            mat = []
            x = get_high_order(X[j], order)
            mat.append(x)
            mat = np.asarray(mat).transpose()
            G += np.matmul(mat, mat.transpose())
            B += Y[j] * mat
        B += reg_lambda
        k = np.random.rand(order)
        w = [k]
        w = np.asarray(w).transpose()
        g = np.matmul(G, w) + B
        d = -g
        epoch = 0
        t = []
        y1 = []
        y2 = []
        while np.linalg.norm(g) >= threshold and epoch <= order:
            epoch += 1
            a = (np.vdot(g, g)) / (np.vdot(d, np.matmul(G, d)))
            w = w + a * d
            g_new = np.matmul(G, w) + B
            beta = np.vdot(g_new, g_new) / np.vdot(g, g)
            g = g_new
            d = -g + beta * d

            t.append(epoch)
            loss = 0
            for j in range(len(X)):
                l = 0
                for k in range(order):
                    l += -w[k] * (X[j] ** k)
                loss += (l - Y[j]) ** 2
            y1.append(math.sqrt(loss / len(X)))
            loss = 0
            for j in range(21):
                l = 0
                for k in range(order):
                    l += -w[k] * (test_x[j] ** k)
                loss += (l - math.sin(2 * math.pi * test_x[j])) ** 2
            y2.append(math.sqrt(loss / 21))
        loss = 0
        w = -w
        return w, t, y1, y2
