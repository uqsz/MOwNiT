import pandas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.linalg import lstsq


# a) Otwórz zbiory breast-cancer-train.dat i breast-cancer-validate.dat
# uzywajac funkcji pd.io.parsers.read csv z biblioteki pandas
labels=pd.read_csv('breast-cancer.labels', header=None)

train_data = pd.read_csv('breast-cancer-train.dat', header=None)
train_data.columns = labels[0].values

validate_data = pd.read_csv('breast-cancer-validate.dat', header=None)
validate_data.columns = labels[0].values


# b) Stwórz histogram i wykres wybranej kolumny danych przy pomocy funkcji hist oraz plot
def show(data, column):
    plt.hist(data[column], bins=10)
    plt.xlabel('Wartość')
    plt.ylabel('Liczba obserwacji')
    plt.title(f'Histogram kolumny {column}')
    plt.show()
    
    plt.plot(data[column],'o')
    plt.xlabel('Indeks obserwacji')
    plt.ylabel('Wartość')
    plt.title(f'Wykres kolumny {column}')
    plt.show()


# c) Stwórz reprezentacje danych zawartych w obu zbiorach dla liniowej i kwadratowej
# metody najmniejszych kwadratów
matrix_linear_train = train_data.iloc[:,2:].values
matrix_linear_validate = validate_data.iloc[:,2:].values

subset = ['radius (mean)', 'perimeter (mean)', 'area (mean)', 'symmetry (mean)']
matrix_quadr_train = np.hstack((train_data[subset], train_data[subset] ** 2))
matrix_quadr_validate = np.hstack((validate_data[subset], validate_data[subset] ** 2))


# d) Stwórz wektor b dla obu zbiorów
train_labels = train_data.iloc[:, 1].values
train_b = np.where(train_labels == 'M', 1, -1)

validate_labels = validate_data.iloc[:, 1].values
validate_b = np.where(validate_labels == 'M', 1, -1)


# e) Znajdz wagi dla liniowej oraz kwadratowej reprezentacji najmniejszych kwadratów
w_train_lin= lstsq(matrix_linear_train, train_b)
w_validate_lin= lstsq(matrix_linear_validate, validate_b)

w_train_quad= lstsq(matrix_quadr_train, train_b)[0]
w_validate_quad= lstsq(matrix_quadr_validate, validate_b)[0]


# f) Oblicz współczynnik uwarunkowania
cond_train_lin = np.linalg.cond(matrix_linear_train)
cond_train_quad = np.linalg.cond(matrix_linear_validate)

# h) Sprawdz jak dobrze otrzymane wagi przewiduja typ nowotworu
p_linear_validate = matrix_linear_validate.dot(w_validate_lin[0])
p_quad_validate = matrix_quadr_validate.dot(w_validate_quad)

predicted_linear = np.where(p_linear_validate > 0, 1, -1)
predicted_quad = np.where(p_quad_validate > 0, 1, -1)

false_positives_linear = np.sum(np.logical_and(predicted_linear > 0, validate_b < 0))
false_negatives_linear = np.sum(np.logical_and(predicted_linear < 0, validate_b > 0))

false_positives_quad = np.sum(np.logical_and(predicted_quad > 0, validate_b < 0))
false_negatives_quad = np.sum(np.logical_and(predicted_quad < 0, validate_b > 0))

# wyniki
show(train_data, 'radius (mean)')


print("Współczynnik uwarunkowania dla reprezentacji liniowej train:", cond_train_lin)
print("Współczynnik uwarunkowania dla reprezentacji kwadratowej train:", cond_train_quad)

print(f"Liczba fałszywie dodatnich dla reprezentacji liniowej: {false_positives_linear}")
print(f"Liczba fałszywie ujemnych dla reprezentacji liniowej: {false_negatives_linear}")
print(f"Liczba fałszywie dodatnich dla reprezentacji kwadratowej: {false_positives_quad}")
print(f"Liczba fałszywie ujemnych dla reprezentacji kwadratowej: {false_negatives_quad}")

























