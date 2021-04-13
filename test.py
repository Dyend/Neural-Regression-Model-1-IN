import my_utility as ut



if __name__ == "__main__":
  inp = "./data/test_x.csv"
  out = "./data/test_y.csv"
  xv = ut.csv_to_matrix(inp)
  yv = ut.csv_to_matrix(out)
  w1,w2 = ut.cargar_pesos()
  zv = ut.snn_ff_old(xv,w1,w2)
  ut.metricas(yv,zv)
  ut.generar_costo(yv,zv)