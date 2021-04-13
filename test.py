import my_utility as ut



if __name__ == "__main__":
  inp = "./data/test_x.csv"
  out = "./data/test_y.csv"
  xv = ut.csv_to_matrix(inp)
  yv = ut.csv_to_matrix(out)
  w1,w2 = ut.cargar_pesos()
  act = ut.snn_ff(xv,w1,w2)
  ut.metricas(yv,act[2])
  ut.generar_costo(yv,act[2])