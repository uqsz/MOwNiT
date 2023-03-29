import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return 1/(1+25*x**2)

def f2(x):
    return np.exp(np.cos(x))

def nodes(a,b,n,f):
    X=np.linspace(-1,1,n)
    Y=f(X)
    return X,Y

def lagrange(Y, B, x):
    n = len(Y)
    L = np.zeros(n)
    for i in range(n):
        L[i] = np.prod((x - Y[np.arange(n)!=i])/(Y[i]-Y[np.arange(n)!=i]))
    return np.sum(B*L)

# wyznaczanie wartości wielomianu
def calc_poly_l(Y,B):
    Tx = np.linspace(-1,1,1000)
    Ty = np.zeros_like(Tx)
    for i in range(len(Tx)):
        Ty[i] = lagrange(Y, B, Tx[i])
    return Tx,Ty

def show(T,A,B,x=""):
    plt.clf()
    plt.title(f"Wielomian interpolacyjny - {x}")
    plt.xlabel("Lata")
    plt.ylabel("Wielkosc populacji")
    plt.figure(figsize=(10,8))
    plt.plot(A[0],A[1],color="blue",)
    plt.plot(T[0],T[1],'o',color="red")
    plt.plot(B[0],B[1],color="orange")
    plt.savefig(f"lab4/pictures/Wielomian interpolacyjny - {x}")
    plt.show()


xy1=nodes(-1,1,13,f1)
xy2=calc_poly_l(xy1[0],xy1[1])
xy3=nodes(-1,1,100,f1)
show(xy1,xy2,xy3)