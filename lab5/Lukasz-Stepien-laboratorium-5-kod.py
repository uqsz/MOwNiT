import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import scipy.integrate as spi

# ZADANIE 1

def vand(Y, n):
    return np.fromfunction(lambda i, j: Y[i.astype(int)]**(n-j-1), (9, n), dtype=float)

def AIC_c(m, n, s):
    k = m+1
    return 2*k+n*np.log(s/n)+2*k*(k+1)/(n-k-1)

true_1990 = 248709873

rok = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980])
populacja = np.array([76212168, 92228496, 106021537, 123202624,
                     132164569, 151325798, 179323175, 203302031, 226542199])

result1 = []
result2 = []

for m in range(0, 7):
    A = vand(rok, m+1)
    wsp, res, rnk, s = lstsq(A, populacja, rcond=None)

    X = np.arange(1890, 1991, 1)
    values = np.polyval(wsp, X)

    temp = abs(true_1990-values[-1])/true_1990
    result1.append(temp)

    X_years = np.arange(1900, 1981, 10)
    values_years = np.polyval(wsp, X_years)

    sqr = np.sum((values_years - populacja) ** 2)
    result2.append(AIC_c(m, 9, sqr))

    # # Wygenerowanie wykresu (odkomentować dla pokazania wykresów)
    # plt.figure()
    # plt.scatter(rok, populacja, label='Dane populacyjne')
    # plt.semilogy(X, wielomian, label=f'Wielomian stopnia {m}')
    # plt.xlabel('Rok')
    # plt.ylabel('Populacja')
    # plt.title(f'Aproksymacja średniokwadratowa punktowa\nStopień wielomianu: {m}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

plt.semilogy(result1, 'o-')
plt.xlabel('Stopien wielomianu aproksymacji')
plt.ylabel('Błąd względny')
plt.title('Błąd względny ekstrapolacji dla roku 1990')
plt.grid(True)
plt.savefig("pictures/Błąd względny ekstrapolacji dla roku 1990", dpi=400)
plt.show()


plt.semilogy(result2, 'o-')
plt.xlabel('Stopien wielomianu aproksymacji')
plt.ylabel('Wspolczynnik AIC_c')
plt.title('Wartosc AIC_c dla danego modelu aproksymacyjnego')
plt.grid(True)
plt.savefig("pictures/Wartosc AIC_c dla danego modelu aproksymacyjnego", dpi=400)
plt.show()

print("{:<8} {:^7} {:^20}".format("Stopien", "Blad", "Wspolczynnik AIC_c"))
for i in range(0, 7):
    print("{:^8} {:^7} {:^20}".format(
        i, ["{:.2%}".format(x) for x in result1][i], round(result2[i], 2)))

idx_1 = np.argmin(result1)
print("Najmniejszy blad dla wielomianu stopnia:", idx_1)

idx_2 = np.argmin(result2)
print("Najmniejsza wartosc AIC_c dla wielomianu stopnia:", idx_2)

plt.show()


# ZADANIE 2

def f(x):
    return np.sqrt(x+1)

a0 = spi.quad(lambda x: f(x) * 1/np.sqrt(1-x**2), -1, 1)[0] / np.pi
a1 = 2*spi.quad(lambda x: f(x) * x/np.sqrt(1-x**2), -1, 1)[0] / np.pi
a2 = 2*spi.quad(lambda x: f(x) * (2*x**2-1)/np.sqrt(1-x**2), -1, 1)[0] / np.pi

def P(x):
    return a0 + a1 * x + a2 * (2*x**2-1)

x_f = np.linspace(-1, 1, 100)
y_f = f(x_f)
x_f = np.linspace(0, 2, 100)

x_P = np.linspace(-1,1 , 100)
y_P = P(x_P)
x_P = np.linspace(0, 2, 100)

plt.plot(x_f, y_f, label='f(x)', color='blue')
plt.plot(x_P, y_P, label='P(x)', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Aproksymacja ciągła funkcji f(x) za pomocą wielomianu Czebyszewa')
plt.grid(True)
plt.savefig("pictures/Aproksymacja ciągła funkcji f(x) za pomocą wielomianu Czebyszewa (metoda aproksymacji ortogonalnej)", dpi=400)
plt.show()
