def main():
  inp = "text_x.csv"
  out = "test_y.csv"
  file_w = 'pesos.npz'
  xv,yv = ut.load_data_txt(inp,out)
  w1,w2 = ut.