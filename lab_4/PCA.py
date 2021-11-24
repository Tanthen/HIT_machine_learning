import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image
import math


def pca(X, k):
    m = X.shape[0]
    mu = np.sum(X, axis=0) / m
    x = X - mu
    cov = np.matmul(x.T, x) / m
    lam, w = np.linalg.eig(cov)
    index = np.argsort(lam)
    w = w[:, index[:-(k + 1):-1]].T
    return w


def generator_data():
    mean1 = [7, 7]
    cov1 = [[2, 2], [2, 3]]
    data = np.random.multivariate_normal(mean1, cov1, 100)

    mean2 = [2, 2]
    cov2 = [[2, 1], [1, 1]]
    data = np.append(data, np.random.multivariate_normal(mean2, cov2, 100), 0)

    return data


def d2():
    data = generator_data()
    w = pca(X=data, k=1)
    print(w.shape)
    val = np.matmul(w, data.T)
    sub = np.matmul(val.T, w)
    w = w[0]
    print(w)
    x0 = np.linspace(0, 10)
    y = w[0] * x0 / w[1]
    fig = plt.figure(figsize=(20, 10))
    sp = fig.add_subplot(1, 1, 1)
    sp.axis([0, 19, 0, 14])
    sp.scatter(data[:, 0], data[:, 1], color='blue')
    sp.scatter(sub[:, 0], sub[:, 1], color='red')
    sp.plot(sub[:, 0], sub[:, 1], color='green')
    plt.show()


def generator_3d():
    data = np.zeros((100, 3))
    for i in range(100):
        x = np.random.random(1) * 10 - 5
        y = np.random.random(1) * 10 - 5
        z = np.sin(np.sqrt(x ** 2 + y ** 2))
        data[i][0] = x
        data[i][1] = y
        data[i][2] = z
    return data


def d3():
    data = generator_3d()
    w = pca(data, 2)
    print(data.shape)
    print(w.shape)
    sub = np.matmul(data, w.T)  # (100, 2)
    sub = np.matmul(sub, w)
    print(w)
    fig4 = plt.figure()
    ax4 = plt.axes(projection='3d')
    print(np.size(data[:, 0]), np.size(data[:, 1]), np.size(data[:, 2]))
    ax4.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.3, c=np.random.random(100),
                s=np.random.randint(20, 40, size=(5, 20)))
    sub0, sub1 = np.meshgrid(sub[:, 0], sub[:, 1])
    plt.show()


def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def get_img():
    img = np.zeros((1777, 2500))
    skip = 0
    for i in range(1777):
        name = str(61853 + i + skip) + "_2019.jpg"
        im = None
        try:
            im = Image.open("/lab_4/data/" + name)
        except FileNotFoundError as e:
            skip += 1
        if im:
            img[i] = np.array(im).flatten()
    return img


def d4():
    data = get_img()
    w = pca(data, 100)
    print(data.shape)
    print(w.shape)
    new_img = np.matmul(data, w.T)
    new_img = np.matmul(new_img, w)
    new_img = new_img.reshape((1777, 50, 50))
    data = data.reshape((1777, 50, 50))
    # im = Image.fromarray(data[0])
    # im = im.convert('L')
    # im.save("/data/image.jpg")
    # print(new_img[0].shape)
    # im = Image.fromarray(np.uint8(new_img[0]))
    # im = im.convert('L')
    # im.save("/data/image2.jpg")
    a = psnr(data[0], new_img[0])
    print(a)


if __name__ == '__main__':
    # d2()
    # d3()
    d4()
