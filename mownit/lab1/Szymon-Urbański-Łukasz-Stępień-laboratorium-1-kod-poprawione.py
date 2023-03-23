from math import tan, log, sqrt
import matplotlib.pyplot as plt
import numpy as np

#Zadanie 1
#Zadanie 1
def f_prim_est1(x, h):
    return (tan(x + h) - tan(x)) / h

def f_prim_est2(x, h):
    return (tan(x + h) - tan(x - h)) / (2 * h)

def f_prim_ex(x):
    return 1 + tan(x) ** 2

def zad1(f, x):
    T = []
    b = f_prim_ex(x)
    for k in range(17):
        h = 10 ** (-k)
        a = f(x, h)
        T.append((log(h, 10), log(abs(a - b), 10)))
    t = list(zip(*T))
    plt.plot(t[0], t[1], 'o')
    plt.title("Wykres zaleznosci bledu od skoku h dla argumentu x=1")
    plt.ylabel("Wartosc bledu bezwzglednego (10^y)")
    plt.xlabel("Wartosc skoku h (10^x)")
    plt.show()
    temp = min(T, key=lambda x: x[1])
    h = 10 ** temp[0]
    print("h_min: ", h)
    print("sqrt(e_mach): ", sqrt(np.finfo(float).eps))

zad1(f_prim_est1, 1)
zad1(f_prim_est2, 1)

#Zadanie 2
def generate1(n):
    T = [0 for _ in range(n)]
    T[0] = np.float32(1 / 3)
    T[1] = np.float32(1 / 12)
    for i in range(2, n):
        T[i] = np.float32(2.25*T[i-1] - 0.5*T[i-2])
    return T

def generate2(n):
    T = [0 for _ in range(n)]
    T[0] = 1 / 3
    T[1] = 1 / 12
    for i in range(2, n):
        T[i] = 2.25*T[i-1] - 0.5*T[i-2]
    return T

def plot(T):
    plt.semilogy(T, 'o')
    plt.xlabel("Indeks wyrazu")
    plt.ylabel("Zalogarytmizowana wartość wyrazu ciągu")
    return plt

def zad2(n1, n2):
    T1 = generate1(n1)
    T2 = generate2(n2)
    T_real = [4 ** (-k) / 3 for k in range(n2)]
    error1 = [abs(T1[i] - T_real[i]) for i in range(n1)]
    error2 = [abs(T2[i] - T_real[i]) for i in range(n2)]

    plt = plot(T1)
    plt.title("Wyznaczone wartości ciągu dla pojedyńczej precyzji")
    plt.xticks([0, 15, 30, 45, 59], [1, 16, 31, 46, 60])
    plt.show()

    plt = plot(T2)
    plt.title("Wyznaczone wartości ciągu dla podwójenj precyzji")
    plt.xticks([0, 50, 100, 150, 200], [1, 51, 101, 151, 201])
    plt.show()

    plt = plot(T_real)
    plt.title("Rzeczywiste wartości ciągu")
    plt.xticks([0, 50, 100, 150, 200], [1, 51, 101, 151, 201])
    plt.show()

    plt = plot(error1)
    plt.title("Wartrości błędu bezwzględnego dla kolejnych wyrazów\n(pojedyńcza precyzja)")
    plt.xticks([0, 15, 30, 45, 59], [1, 16, 31, 46, 60])
    plt.ylabel("Zlogarytmizowana wartość błędu")
    plt.show()

    plt = plot(error2)
    plt.title("Wartrości błędu bezwzględnego dla kolejnych wyrazów\n(podwójna precyzja)")
    plt.xticks([0, 50, 100, 150, 200], [1, 51, 101, 151, 201])
    plt.ylabel("Zlogarytmizowana wartość błędu")
    plt.show()

zad2(60, 225)