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