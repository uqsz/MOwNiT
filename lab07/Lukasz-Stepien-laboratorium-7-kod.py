import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import quad_vec

# Definicje funkcji do całkowania
def f1(x):
    return 4 / (1 + x**2)

def f2(x):
    return np.sqrt(x)*np.log(x)

def f3(x):
    return (1/((x-0.3)**2+0.001))+(1/((x-0.9)**2+0.004))-6

def count_res(a,x0):
    return (1/np.sqrt(a))*(np.arctan((1-x0)/np.sqrt(a))+np.arctan(x0/np.sqrt(a)))


def integrate_all(a,b,f,ex_res,title,cnt):
    # Lista przechowująca wartości bezwzględnych błędów względnych dla różnych metod
    errors_trapezoid = []
    errors_simpson = []
    errors_rectangle = []
    errors_gl = []
    errors_adaptive_trapezoid = []
    errors_adaptive_gk = []
    
    # Lista przechowująca ilość węzłów użytych w każdej iteracji
    num_nodes = []
    neval_adaptive_trapezoid = []
    neval_adaptive_gk = []
    
    # Epsilon
    eps=1e-20
    
    for m in range(1,25):
        # Ilość węzłów w aktualnej iteracji
        n = 2 ** m + 1
        num_nodes.append(n)
    
        # Metoda złożonych prostokątów
        x_rect = np.linspace(a, b, n+1)
        result_rect = np.sum(f(x_rect[:-1]) * (b-a)/n)
        error_rect = np.abs((result_rect - ex_res)/ex_res+eps)
        errors_rectangle.append(error_rect)
    
        # Metoda złożonych trapezów
        x_trap = np.linspace(a, b, n)
        result_trap = integrate.trapz(f(x_trap), x_trap)
        error_trap = np.abs((result_trap - ex_res)/ex_res+eps)
        errors_trapezoid.append(error_trap)
        
        # Metoda Simpsona
        x_simp = np.linspace(a, b, n)
        result_simp = integrate.simps(f(x_simp), x_simp)
        error_simp = np.abs((result_simp - ex_res)/ex_res+eps)
        errors_simpson.append(error_simp)
        
        if m<15:
            # Metoda Gaussa-Legendre'a
            nodes, weights = np.polynomial.legendre.leggauss(n)
            approx_value = np.dot(weights, f((b-a)/2 * nodes + (a+b)/2))/2
            error = np.abs((ex_res - approx_value)/ex_res+eps)
            errors_gl.append(error)
    
    for i in range(29):
        # Metoda adaptacyjna trapezów
        result, abserr, info = quad_vec(f, a, b, epsabs=10**(-i/2), full_output=True, quadrature='trapezoid')
        errors_adaptive_trapezoid.append(abs((result-ex_res)/ex_res)+eps)
        neval_adaptive_trapezoid.append(info.neval)
        
        # Metoda adaptacyjna Gaussa-Kronroda
        result, abserr, info = quad_vec(f, a, b, epsabs=10**(-i/2), full_output=True, quadrature='gk21')
        errors_adaptive_gk.append(abs((result-ex_res)/ex_res)+eps)
        neval_adaptive_gk.append(info.neval)
        
    # Wykres
    plt.plot(neval_adaptive_trapezoid,errors_adaptive_trapezoid, '-o',label='Adaptive trapezoid')
    plt.plot(neval_adaptive_gk,errors_adaptive_gk, '-o',label='Adaptive GK')
    plt.plot(num_nodes, errors_trapezoid, 'o-', label='Trapezoid')
    plt.plot(num_nodes, errors_rectangle, 'o-', label='Rectangle')
    plt.plot(num_nodes, errors_simpson, 'o-', label='Simpson')
    plt.plot(np.array(num_nodes[0:14])*2, errors_gl, 'o-', label='Gauss-Legendre')
    plt.title(title)
    plt.ylim([min(min(errors_adaptive_gk),1e-13),1e2])
    plt.xlabel('Liczba ewaluacji funkcji n')
    plt.ylabel('Błąd względny')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='upper right',fontsize='small')
    plt.grid(True)
    plt.savefig(f"pictures/Blad wzgledny dla roznych metod{cnt}", dpi=350)
    
    plt.show()

# Funkcja 1
a = 0
b = 1
cnt=0
ex_res=np.pi
title=r"Obliczana całka: $\int_{0}^{1} \frac{4}{1 + x^2} dx$"

integrate_all(a,b,f1,ex_res,title,cnt)


# Funkcja 2
a = np.finfo(float).eps
b = 1
cnt+=1
ex_res=-4/9
title=r"Obliczana całka: $\int_{0}^{1} \sqrt{x}\log(x) dx$ "

integrate_all(a,b,f2,ex_res,title,cnt)


# Funkcja 3
a = 0
b = 1
cnt+=1
ex_res=count_res(0.001,0.3)+count_res(0.004,0.9)-6
title=r"Obliczana całka: $\int_{0}^{1} (\frac{1}{(x-0.3)^2+a}+\frac{1}{(x-0.9)^2+b}-6) dx$ "

integrate_all(a,b,f3,ex_res,title,cnt)

























    
    
    