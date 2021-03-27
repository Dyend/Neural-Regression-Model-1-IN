import numpy as np




def p_inversa(a1,ye,hn,C):
  yh = np.dot(ye,a1.T)
  ai = np.dot(a1,a1.T)+ np.eye(hn)/C
  p_inv = np.linalg.pinv(ai)
  w2 = np.dot(yh,p_inv)
  return w2



def train_snn(xe,ye,hn,C):
  n0 = xe.shape[0]
  w1 = ut.iniW(hn,n0)
  z = np.dot(w1,xe)
  a1 = 1/(1+np.exp(-z)) 
  w2 = p_inversa(a1,y,hn,C)
  return (w1,w2) #w1 capa oculta w2 capa de salida 


def main():
  inp = "train_x.csv""
  out = "train_y.csv"
  hn,C = ut.load_data_txt(inp,out)
  w1,w2 = train_snn(xe,ye,hn,C)
  ut.save_w_npy(w1,w2)

  
