import numpy as np
import my_utility as ut


def forward_dl(x, w):
    L = len(w) - 1
    n_fw = int(L/2)
    for i in range(n_fw):
        x = ut.act_sigmoid(np.dot(w[i], x))
    zv = ut.softmax(np.dot(w[L], x))

    return zv


def main():
    xv = ut.load_data_csv("./data/test_x.csv")
    yv = ut.load_data_csv("./data/test_y.csv")
    W = ut.load_w_dl("./data/w_dl.npz")
    zv = forward_dl(xv,W)
    Fsc = ut.metricas(yv,zv)


if __name__ == "__main__":
    main()
