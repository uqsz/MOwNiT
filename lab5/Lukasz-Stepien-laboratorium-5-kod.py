import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebval

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

# Przeprowadzenie aproksymacji średniokwadratowej punktowej dla różnych stopni m
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
    return np.sqrt(x)


x = np.linspace(0, 2, 1000)
y = f(x)

deg = 2
coeffs = chebfit(x, y, deg)

x_aprox = np.linspace(0, 2, 1000)
y_aprox = chebval(x_aprox, coeffs)

plt.plot(x, y, label='f(x) = sqrt(x)')
plt.plot(x_aprox, y_aprox, label='Aproksymacja')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Aproksymacja funkcji sqrt(x) wielomianem Czebyszewa')
plt.grid(True)
plt.savefig(
    "pictures/Aproksymacja funkcji sqrt(x) wielomianem Czebyszewa", dpi=400)
plt.show()
