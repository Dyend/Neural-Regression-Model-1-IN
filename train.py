import numpy as np
import my_utility as ut


def train_softmax(x, y, param):
    w = ut.randW(y.shape[0], x.shape[0])
    costos = []
    for iter in range(1, param[0]):
        gradW, costo = ut.softmax_grad(x, y, w, param[2])
        costos.append(costo)

        if (iter % 100) == 0:
            print(f'Costo iteración {iter}: {costo}')

        w -= param[1] * gradW

    return (w, costos)


def get_miniBatch(i, x, bsize):
    bsize = int(bsize)
    start_idx = i * bsize
    xe = x[:, start_idx: start_idx + bsize]
    return (xe)


def train_dae(x,W,numBatch,BatchSize,mu):
    for i in range(numBatch):        
        xe   = get_miniBatch(i,x,BatchSize)
        Act  = ut.forward_dae(xe,W)
        gW   = ut.grad_bp_dae(Act,W)
        W    = ut.updW_sgd(W,gW,mu)
    return(W)


def train_dl(x, param):
    W = ut.iniW(x.shape[0], param[3:])
    batchId = np.int16(np.floor(x.shape[1] / param[1]))
    tau = param[0] / param[2]
    for Iter in range(param[2]):
        xe = x[:, np.random.permutation(x.shape[1])]
        mu = param[0] / (1 + tau * Iter)
        W = train_dae(xe, W, batchId, param[1], mu)
    return(W)


def main():
    par_dae, par_sft = ut.load_config()
    xe = ut.load_data_csv("./data/train_x.csv")
    ye = ut.load_data_csv("./data/train_y.csv")
    W = train_dl(xe, par_dae)
    Xr = ut.encoder(xe, W)
    Ws, cost = train_softmax(Xr,ye,par_sft)
    ut.save_w_dl(W, Ws, "./data/w_dl.npz", cost, "./data/costo_sofmax.csv")


if __name__ == "__main__":
    main()
