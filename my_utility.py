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
  mu = config[3]
  maxIter = config[4]
  return p, hn, C, mu, maxIter

def csv_to_matrix(ruta):
  file = open(ruta)
  numpy_array = numpy.loadtxt(file, delimiter=",")
  return numpy_array

def iniW(hn,n0):
  r = math.sqrt( 6 / (hn+n0) )
  matrix=[]
  for i in range(0, int(hn)):
      row=[]
      for j in range(0, int(n0)):
          row.append(random.random() * 2 * r - r)
      matrix.append(row)
  matrix = numpy.array(matrix)
  return matrix




def generar_pesos(w1,w2):
  np.savez_compressed('./data/pesos.npz', matrixw1=w1, matrixw2=w2)

def cargar_pesos():
  b = np.load('./data/pesos.npz')
  return b['matrixw1'],b['matrixw2']

def snn_ff_old(xv,w1,w2):
  zv = np.dot(w1,xv)
  a1 = (1/(1+np.exp(-zv)))
  z2 = np.dot(w2,a1)
  return z2

def snn_ff(xv,w1,w2):
  a = []
  zv = np.dot(w1,xv)
  a1 = (1/(1+np.exp(-zv)))
  z2 = np.dot(w2,a1)
  a2 = (1/(1+np.exp(-z2)))
  a.append(xv)
  a.append(a1)
  a.append(a2)
  return a

def derivate_act(a):
  z = numpy.log((1/a)-1)
  derivate_act = (-(np.exp(-z)/np.square((1+np.exp(-z)))))
  return derivate_act

def snn_bw(act, ye, w1, w2, mu):

  e = act[2] - ye
  Cost = np.mean(e**2)
  dOut = e * derivate_act(act[2])
  gradW2 = np.dot(dOut, act[1].T)
  dHidden = np.dot(w2.T, dOut) * derivate_act(act[1])
  gradW1 = np.dot(dHidden, act[0].T)
  w2 = w2 - mu * gradW2
  w1 = w1 - mu * gradW1

  return w1, w2, Cost

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
  yv = yv.reshape(-1, 1)
  zv = zv.reshape(-1, 1)
  costos_data = np.hstack((yv,zv))
  numpy.savetxt(ruta, costos_data, delimiter=",")