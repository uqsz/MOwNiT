import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import quad_vec

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

def eps(t,k,x_0):
    return abs(t[k]-x_0)

def r(t,k,x_0):
    return np.log(eps(t,k,x_0)/eps(t,k+1,x_0))/np.log(eps(t,k-1,x_0)/eps(t,k,x_0))
    
x0=3

def err(x):
    x0=2
    return abs((x-x0)/x0)
    

t1=iterate(x0,g1)
print(r(t1,8,x0))
t1=err(np.array(t1))


t2=iterate(x0,g2)
print(r(t2,8,x0))
t2=err(np.array(t2))

t3=iterate(x0,g3)
print(r(t3,8,x0))
t3=err(np.array(t3))

t4=iterate(x0,g4)
print(r(t4,6,x0))
t4=err(np.array(t4))

print(t1,t2,t3,t4)

plt.semilogy(t1, label='t1')
plt.semilogy(t2, label='t2')
plt.semilogy(t3, label='t3')
plt.semilogy(t4, label='t4')
plt.legend()
plt.show()

