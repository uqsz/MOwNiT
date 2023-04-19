import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Definicja funkcji do całkowania
def f(x):
    return 4 / (1 + x**2)

# Przedział całkowania
a = 0
b = 1

# Lista przechowująca wartości bezwzględnych błędów względnych dla różnych metod
errors_trapezoid = []
errors_simpson = []
errors_rectangle = []

# Lista przechowująca ilość węzłów użytych w każdej iteracji
num_nodes = []


for m in range(1,25):
    n = 2 ** m + 1  # Ilość węzłów w aktualnej iteracji
    num_nodes.append(m)

    # Metoda złożonych prostokątów
    x_rect = np.linspace(a, b, n+1)
    result_rect = np.sum(f(x_rect[:-1]) * (b-a)/n)
    error_rect = np.abs((result_rect - np.pi)/np.pi)
    errors_rectangle.append(error_rect)

    # Metoda złożonych trapezów
    x_trap = np.linspace(a, b, n)
    result_trap = integrate.trapz(f(x_trap), x_trap)
    error_trap = np.abs((result_trap - np.pi)/np.pi)
    errors_trapezoid.append(error_trap)
    
    # Metoda Simpsona
    x_simp = np.linspace(a, b, n)
    result_simp = integrate.simps(f(x_simp), x_simp)
    error_simp = np.abs((result_simp - np.pi)/np.pi)
    errors_simpson.append(error_simp)


def h(m):
    return 1/(2**m+1)

print("Wartosc h_min dla metody trapezow:", h(22))
print("Wartosc h_min dla metody simpsona:", h(6))
print("Wartosc h_min z lab1:  ", 1e-07,"\n")

m1 = 5
m2 = 20
print(f"Empiryczne rzędy zbieżności dla m1={m1} m2={m2}:")
r = round(np.log(errors_rectangle[m2]/errors_rectangle[m1])/np.log(h(m2)/h(m1)),4)
t = round(np.log(errors_trapezoid[m2]/errors_trapezoid[m1])/np.log(h(m2)/h(m1)),4)
m1 = 1
m2 = 6
s = round(np.log(errors_simpson[m2]/errors_simpson[m1])/np.log(h(m2)/h(m1)),4)


print(f"Prostokaty: {r}")
print(f"Trapezy: {t}")
print(f"Empiryczne rzędy zbieżności dla m1={m1} m2={m2}:")
print(f"Simpson: {s}")



#przeksztalcona funkcja z zmienionymi granicami calkowania na -1,1
def f1(t):
    return 2/(1+((t+1)/2)**2)

a = -1
b = 1
n_list = np.arange(1, 24)

error_list = []

for n in n_list:
    nodes, weights = np.polynomial.legendre.leggauss(n)
    approx_value = np.dot(weights, f1(nodes))
    error = np.abs((np.pi - approx_value)/np.pi)
    error_list.append(error)

plt.plot(n_list, error_list, 'o-', label='Gauss-Legendre')
plt.plot(num_nodes, errors_trapezoid, 'o-', label='Trapezoid')
plt.plot(num_nodes, errors_rectangle, 'o-', label='Rectangle')
plt.plot(num_nodes, errors_simpson, 'o-', label='Simpson')
plt.title('Blad wzgledny dla roznych metod')
plt.xlabel('wartosc m')
plt.ylabel('Błąd względny względem $\pi$')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig("pictures/Blad wzgledny dla roznych metod", dpi=350)

plt.show()

























    
    
    