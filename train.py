import numpy as np

def train_snn(xe,ye,hn,C):
  n0 = xe.shape[0]
  w1 = ut.iniW(hn,n0)
  z = np.dot(w1,xe)
  a1 = 1/(1+np.exp(-z))
  w2 = p_inversa(a1,y,hn,C)
  return (w1,w2)