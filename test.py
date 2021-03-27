def main():
  inp = "./data/text_x.csv"
  out = "./data/test_y.csv"
  xv = ut.csv_to_matrix(inp)
  yv = ut.csv_to_matrix(out)
  w1,w2 = ut.cargar_pesos()
  zv = ut.snn_ff(xv,w1,w2)
  ut.metricas(yv,zv)
