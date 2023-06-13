import numpy as np
import matplotlib.pyplot as plt
import scipy

# Rosenbrock function


def f(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# gradient of Rosenbrock function


def gradient(x):
    grad = np.zeros(2)
    grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    grad[1] = 200 * (x[1] - x[0]**2)
    return grad

# hessian of Rosenbrock function


def hessian(x):
    hessian = np.zeros((2, 2))
    hessian[0, 0] = 1200 * x[0]**2 - 400 * x[1] + 2
    hessian[0, 1] = -400 * x[0]
    hessian[1, 0] = -400 * x[0]
    hessian[1, 1] = 200
    return hessian

# newton method for steepest descent


def newton_method_alpha(x):
    def g(alpha):
        return f(x - alpha * gradient(x))
    alpha = 1
    for _ in range(20):
        derivative_1 = scipy.misc.derivative(g, alpha, dx=1e-4, n=1)
        derivative_2 = scipy.misc.derivative(g, alpha, dx=1e-4, n=2)
        if derivative_2 == 0:
            break
        alpha -= derivative_1 / derivative_2
    return alpha

# steepest descent


def steepest_descent(x0, N):
    x = x0
    T = [f(x)]
    for _ in range(N):
        alpha = newton_method_alpha(x)
        x = x - alpha * gradient(x)
        T.append(f(x))
    return T


# newton


def newton_method(x0, N):
    x = x0
    T = [f(x)]
    for _ in range(N):
        x = x - np.linalg.inv(hessian(x)) @ gradient(x)
        T.append(f(x)+1e-35)
    return T

# results


def show(func, N, s):
    x0 = np.array([-1, 1])
    x1 = np.array([0, 1])
    x2 = np.array([2, 1])

    X = np.arange(0, N+1)

    Y1 = func(x0, N)
    Y2 = func(x1, N)
    Y3 = func(x2, N)

    plt.semilogy(X, Y1, "-o", label="x0 = (-1, 1)")
    plt.semilogy(X, Y2, "-o", label="x0 = (0, 1)")
    plt.semilogy(X, Y3, "-o", label="x0 = (2, 1)")

    plt.grid(True)
    plt.xlabel("Iteracje")
    plt.ylabel("Zminimalizowana wartość funkcji Rosenbrocka")
    plt.title(f"Metoda {s} dla różnych x0")
    plt.legend()
    plt.savefig(f"lab11/pictures/Metoda {s} dla różnych x0", dpi=350)
    plt.clf()
    # plt.show()


show(steepest_descent, 10, "największego spadku")
show(newton_method, 10, "Newtona")
