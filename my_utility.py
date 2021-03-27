import csv
import numpy

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