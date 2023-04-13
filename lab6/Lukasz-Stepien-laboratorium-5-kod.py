import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import scipy.integrate as spi

def f(x):
    return 4/(1+x**2)

def rect(f,a,b,n):
    h=abs(a-b)/n
    x=np.linspace(a, b, n)
    y=f(x)
    res=h*sum(y)
    return res

print(rect(f,0,1,10**5))

def trapeze(f,a,b,n):
    h=abs(a-b)/n
    x=np.linspace(a, b, n)
    y=f(x)
    
    
    
    
    
    