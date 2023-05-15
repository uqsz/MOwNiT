import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import quad_vec
import math

# ZADANIE 1

def g1(x):
    return (x**2+2)*(1/3)

def g2(x):
    return np.sqrt(3*x-2)

def g3(x):
    return 3-(2/x)

def g4(x):
    return (x**2-2)/(2*x-3)

def iterate(x0,g):
    t=[0 for _ in range(10)]
    t[0]=x0
    for i in range(9):
        t[i+1]=g(t[i])
    return t

def eps_k(t,k,x_0):
    return abs(t[k]-x_0)

def r(t,k,x_0):
    return np.log(eps_k(t,k,x_0)/eps_k(t,k+1,x_0))/np.log(eps_k(t,k-1,x_0)/eps_k(t,k,x_0))
    

def err(x):
    x0=2
    return abs((x-x0)/x0)
    
eps=1e-20
x0=3

print("Eksprymentalne rzędy zbieżnosci:")

t1=iterate(x0,g1)
print("g1 ->", round(r(t1,8,x0),4))
t1=err(np.array(t1))

t2=iterate(x0,g2)
print("g2 ->",round(r(t2,8,x0),4))
t2=err(np.array(t2))

t3=iterate(x0,g3)
print("g3 ->",round(r(t3,8,x0),4))
t3=err(np.array(t3))

t4=iterate(x0,g4)
print("g4 ->",round(r(t4,6,x0),4))
t4=err(np.array(t4))+eps

plt.semilogy(t1,'-o',label=r'$g1(x)=\frac{1}{3}(x^2+2)$')
plt.semilogy(t2,'-o', label=r'$g2(x)=\sqrt{3x-2}$')
plt.semilogy(t3,'-o', label=r'$g3(x)=3-\frac{2}{x}$')
plt.semilogy(t4,'-o', label=r'$g4(x)=\frac{x^2-2}{2x-3}$')
plt.legend(fontsize='large')
plt.title("Błąd względny w każdej iteracji dla wszystkich funkcji g")
plt.xlabel('Liczba iteracji i')
plt.ylabel('Błąd względny')
#plt.savefig(f"pictures/Błąd względny w każdej iteracji dla wszystkich funkcji g", dpi=350)
plt.show()

plt.semilogy(t2,'-o', label=r'$g2(x)=\sqrt{3x-2}$')
plt.semilogy(t3,'-o', label=r'$g3(x)=3-\frac{2}{x}$')
plt.semilogy(t4,'-o', label=r'$g4(x)=\frac{x^2-2}{2x-3}$')
plt.legend(fontsize='large')
plt.title("Błąd względny w każdej iteracji dla zbieżnych funkcji g ")
plt.xlabel('Liczba iteracji i')
plt.ylabel('Błąd względny')
#plt.savefig(f"pictures/Błąd względny w każdej iteracji dla zbieżnych funkcji g", dpi=350)

plt.show()

# ZADANIE 2

def f1(x):
    return x**3-2*x-5

def f2(x):
    return np.exp(-x)-x

def f3(x):
    return x*np.sin(x)-1

def der(f,x):
    h=1e-8
    return (f(x+h)-f(x))/h

def iterate_newton(x1,f,b):
    tol=1/2**b
    cnt=0
    while True:
        x0=x1
        x1=x0-(f(x0)/der(f,x0))
        cnt+=1
        if abs(x1-x0)==np.finfo('float64').eps:
            return x1,cnt,"MACHINE"
        if abs(x1-x0)<=tol:
            return x1,cnt
    return x1,cnt


a1=round(iterate_newton(1,f1,4)[0],4)
a2=round(iterate_newton(5,f2,4)[0],4)
a3=round(iterate_newton(3,f3,4)[0],4)

print("Przyblizenia pierwiastków z dokładnoscia 4 bitów:")
print("f1 ->",iterate_newton(1,f1,4))
print("f2 ->",iterate_newton(5,f2,4))
print("f3 ->",iterate_newton(3,f3,4))

print("\nPrzyblizenia pierwiastków z dokładnoscia 24 bitów:")
print("f1 ->",iterate_newton(a1,f1,24))
print("f2 ->",iterate_newton(a2,f2,24))
print("f3 ->",iterate_newton(a3,f3,24))


print("\nPrzyblizenia pierwiastków z dokładnoscia 53 bitów:")
print("f1 ->",iterate_newton(a1,f1,53))
print("f2 ->",iterate_newton(a2,f2,53))
print("f3 ->",iterate_newton(a3,f3,53))


# ZADANIE 3

def f(x):
    return np.array([x[0]**2 + x[1]**2 - 1, x[0]**2 - x[1]])

def jacobian(x):
    return np.array([[2*x[0], 2*x[1]], [2*x[0], -1]])

def newton_method(f, jacobian, x0, tol):
    while True:
        delta = np.linalg.solve(jacobian(x0), -f(x0))
        x1 = x0 + delta
        if np.linalg.norm(delta) < tol:
            return x1
        x0 = x1

x2=np.sqrt(5)/2-0.5
x1=np.sqrt(x2)

x0 = np.array([1, 1])
tol = 1e-16
result = newton_method(f, jacobian, x0, tol)

err1=abs((result[0]-x1)/x1)
err2=abs((result[1]-x2)/x2)

print(f"\nWynik: x1={np.round(result[0],16)}, x2={np.round(result[1],16)}")
print(f"Błędy względne: x1:{err1}, x2:{err2}")









































