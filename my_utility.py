import csv
import numpy
import random
import math
import numpy as np


def get_config():
  config = csv_to_matrix("./data/config.csv")
  p = config[0]
  hn = config[1]
  C = config[2]
  return p,hn,C

def csv_to_matrix(ruta):
  file = open(ruta)
  numpy_array = numpy.loadtxt(file, delimiter=",")
  return numpy_array

def iniW(hn,n0):
  r = math.sqrt( 6 / (hn+n0) )
  matrix=[]
  for i in range(0, int(hn)):
      row=[]
      for j in range(0, n0):
          row.append(random.random() * 2 * r - r)
      matrix.append(row)
  return matrix




def generar_pesos(w1,w2):
  np.savez_compressed('./data/pesos.npz', matrixw1=w1, matrixw2=w2)

def cargar_pesos():
  b = np.load('./data/pesos.npz')
  return b['matrixw1'],b['matrixw2']

def snn_ff(xv,w1,w2):
  zv = np.dot(w1,xv)
  a1 = (1/(1+np.exp(-zv)))
  a2 = np.dot(w2,a1)
  return a2

def metricas(yv,zv):

  mae = abs(yv - zv).mean()
  mse = (np.square(yv - zv)).mean()
  rmse = math.sqrt(mse)
  r2 = 1-((yv - zv).var()/yv.var())
  print("MAE:",mae)
  print("MSE:",mse)
  print("RMSE:",rmse)
  print("R2:",r2)
  generar_metricas(mae,rmse,r2)

def generar_metricas(mae,rmse,r2):
  ruta = './data/metrica.csv'
  metrica_data = [mae,rmse,r2]
  print(metrica_data)
  file = open(ruta, "w+")
  numpy.savetxt(ruta, metrica_data, delimiter=",")

def generar_costo(yv,zv):
  ruta = './data/costo.csv'
  file = open(ruta, "w+")
  costos_data = []
  costos_data.append(yv)
  costos_data.append(zv)
  numpy.savetxt(ruta, costos_data, delimiter=",")