import csv

with open('./data/x_input.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        print(', '.join(row))



def normalizer():
    max = data.max()
    min = data.min()
    a = 0.01
    b = 0.99
    input_normalizado = map(lambda: ((value - min) / (max - min)) * (b - a) + a, data)

if __name__ == "__main__":
    normalizer()