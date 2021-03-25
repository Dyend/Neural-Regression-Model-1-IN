import csv
import numpy
from random import randint

def csv_to_matrix(ruta):
    file = open(ruta)
    numpy_array = numpy.loadtxt(file, delimiter=",")
    return numpy_array

def random_reorder(data):
    print('reordenando')
    try:
        len_columna = len(data[0]) - 1
        """ Swap las columnas de la matriz cantidad de columna * 2 veces"""
        for i in range(1, len_columna * 2):
            random_columna1 = randint(0, len_columna)
            random_columna2 = randint(0, len_columna)
            data[:,[random_columna1, random_columna2]] = data[:,[random_columna2, random_columna1]]
    except:
        numpy.random.shuffle(data)
    return data

def normalizer(data):
    
    a = 0.01
    b = 0.99
    
    try:
        """ iterando en las filas de la matriz"""
        normalized_data = []
        for fila in data:
            max = numpy.amax(fila)
            min = numpy.amin(fila)
            fila_normalizada = list(map(lambda value: ((value - min) / (max - min)) * (b - a) + a, fila))
            normalized_data.append(fila_normalizada)
            #print(list(fila_normalizada))
        normalized_data = numpy.array(normalized_data)
    except:
        max = numpy.amax(data)
        min = numpy.amin(data)
        normalized_data = list(map(lambda value: ((value - min) / (max - min)) * (b - a) + a, data))
        normalized_data = numpy.array(normalized_data)
    return normalized_data

if __name__ == "__main__":
    result_x = csv_to_matrix("./data/x_input.csv")
    result_y = csv_to_matrix("./data/y_output.csv")
    result_x = random_reorder(result_x)
    result_y = random_reorder(result_y)
    print(result_y)
    result_x = normalizer(result_x)
    result_y = normalizer(result_y)
    print(result_y)