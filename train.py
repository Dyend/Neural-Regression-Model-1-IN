import numpy as np
import my_utility as ut


def p_inversa(a1,ye,hn,C):
  yh = np.dot(ye,a1.T)
  ai = np.dot(a1,a1.T)+ np.eye(int(hn))/C
  p_inv = np.linalg.pinv(ai)
  w2 = np.dot(yh,p_inv)
  return w2



def train_snn_old(xe,ye,hn,C):
  n0 = xe.shape[0]
  w1 = ut.iniW(hn,n0)
  z = np.dot(w1,xe)
  a1 = 1/(1+np.exp(-z)) 
  w2 = p_inversa(a1, ye, hn, C)
  return (w1,w2) #w1 capa oculta w2 capa de salida 

def train_snn(xe, ye, nh, mu, MaxIter):
  n0 = xe.shape[0]
  w1 = ut.iniW(nh,n0)
  w2 = ut.iniW(1,nh)
  mse = []
  Mse_l = []
  for i in range(int(MaxIter)):
    act = ut.snn_ff(xe, w1, w2)
    w1, w2, cost = ut.snn_bw(act, ye, w1, w2, mu)
    print(cost)
    mse.append(cost)
  return w1, w2, mse


if __name__ == "__main__":
  inp = "./data/train_x.csv"
  out = "./data/train_y.csv"
  p,hn,C, mu, maxIter = ut.get_config()
  xe = ut.csv_to_matrix(inp)
  ye = ut.csv_to_matrix(out)
  #w1,w2 = train_snn_old(xe,ye,hn,C)
  w1, w2, mse = train_snn(xe,ye,hn, mu, maxIter)
  ut.generar_pesos(w1,w2)
  ut.generar_mse(mse)
  
