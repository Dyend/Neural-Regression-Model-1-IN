import numpy as np
import my_utility as ut



def p_inversa(a1,ye,hn,C):
  yh = np.dot(ye,a1.T)
  ai = np.dot(a1,a1.T)+ np.eye(int(hn))/C
  p_inv = np.linalg.pinv(ai)
  w2 = np.dot(yh,p_inv)
  return w2



def train_snn(xe,ye,hn,C):
  n0 = xe.shape[0]
  w1 = ut.iniW(hn,n0)
  z = np.dot(w1,xe)
  a1 = 1/(1+np.exp(-z)) 
  w2 = p_inversa(a1, ye, hn, C)
  return (w1,w2) #w1 capa oculta w2 capa de salida 


if __name__ == "__main__":
  inp = "./data/train_x.csv"
  out = "./data/train_y.csv"
  p,hn,C = ut.get_config()
  xe = ut.csv_to_matrix(inp)
  ye = ut.csv_to_matrix(out)
  w1,w2 = train_snn(xe,ye,hn,C)
  print(w1)
  print(w2)
  #ut.save_w_npy(w1,w2)

  
