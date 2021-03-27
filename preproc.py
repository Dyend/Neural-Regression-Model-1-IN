import csv
import numpy
import my_utility as ut
from random import randint


def random_reorder(data_x, data_y):
    print('reordenando')
    try:
        len_columna_x = len(data_x[0]) - 1
        """ Swap las columnas de la matriz cantidad de columna * 2 veces"""
        for i in range(1, len_columna_x * 2):
            random_columna1 = randint(0, len_columna_x)
            random_columna2 = randint(0, len_columna_x)
            data_x[:,[random_columna1, random_columna2]] = data_x[:,[random_columna2, random_columna1]]
            data_y[[random_columna1, random_columna2]] = data_y[[random_columna2, random_columna1]]
    except:
      print("error reorder")
    return data_x, data_y

def normalizer(data):
    print('normalizando')
    a = 0.01
    b = 0.99
    
    try:
        """ iterando en las filas de la matriz"""
        if (len(data.shape) == 2):
          normalized_data = []
          for fila in data:
              max = numpy.amax(fila)
              min = numpy.amin(fila)
              fila_normalizada = list(map(lambda value: ((value - min) / (max - min)) * (b - a) + a, fila))
              normalized_data.append(fila_normalizada)
              #print(list(fila_normalizada))
          normalized_data = numpy.array(normalized_data)
        elif (len(data.shape) == 1):
          max = numpy.amax(data)
          min = numpy.amin(data)
          normalized_data = list(map(lambda value: ((value - min) / (max - min)) * (b - a) + a, data))
          normalized_data = numpy.array(normalized_data)
    except:
       print("error en normalizer")
    return normalized_data
 

def generar_train(data, porcentaje_training,ruta):
    print('generando archivo ' + ruta)
    if (len(data.shape) == 2):
      data.shape
      cantidad_training = int(data.shape[1] * porcentaje_training)
      data = data[0:data.shape[0], 0:cantidad_training]
    elif (len(data.shape) == 1):
      cantidad_training = int(data.shape[0] * porcentaje_training)
      data = data[0:cantidad_training]
    try:
        file = open(ruta, "w+")
        numpy.savetxt(ruta, data, delimiter=",")
    except:
        print("error en generar_train")

def generar_test(data, ruta):
    print('generando archivo ' + ruta)
    try:
      file = open(ruta, "w+")
      numpy.savetxt(ruta, data, delimiter=",")
    except:
        print("error en generar_test")

if __name__ == "__main__":
    result_x = ut.csv_to_matrix("./data/x_input.csv")
    result_y = ut.csv_to_matrix("./data/y_output.csv")
    p,hn,C = ut.get_config()
    result_x, result_y = random_reorder(result_x, result_y)
    result_x = normalizer(result_x)
    result_y = normalizer(result_y)
    generar_train(result_x, p, "./data/train_x.csv")
    generar_train(result_y, p, "./data/train_y.csv")
    generar_test(result_x, "./data/test_x.csv")
    generar_test(result_y, "./data/test_y.csv")