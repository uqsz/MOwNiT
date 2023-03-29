import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

#funkcja nr 1
def f1(x): 
    return 1/(1+25*x**2)

#funkcja nr 2
def f2(x):
    return np.exp(np.cos(x))

#wyznaczenie równoodległych wezlow interpolacji
def nodes(a,b,n,f):
    X=np.linspace(a,b,n)
    Y=f(X)
    return X,Y

#wyznaczenie kąta theta dla wezlow Czebyszewa
def th(i,n):
    return ((2*i+1)/(2*(n+1)))*np.pi

#wyznaczenie równoodległych wezlow interpolacji Czebyszewa
def nodes_cz(a,b,n,f):
    X=np.fromfunction(lambda i: np.cos(th(i,n)) , (n,), dtype=float)
    Y=f(X)
    return X,Y

#wyznaczenie równoodległych wezlow interpolacji Czebyszewa i ich transformacja
def nodes_cz_t(a,b,n,f):
    X=np.fromfunction(lambda i: np.cos(th(i,n)) , (n,), dtype=float)
    X=a+(b-a)*(X+1)/2
    Y=f(X)
    return X,Y

def lagrange(X, Y, x):
    n = len(X)
    L = np.zeros(n)
    for i in range(n):
        L[i] = np.prod((x - X[np.arange(n)!=i])/(X[i]-X[np.arange(n)!=i]))
    return np.sum(Y*L)

#wyznaczenie wielomianu Lagrange'a
def calc_poly_l(X,Y,a,b,n,is_rand=False):
    if is_rand:
        np.random.seed(1234)
        Tx=np.random.uniform(a,b,n)
    else:
        Tx = np.linspace(a,b,n)
    Ty = np.zeros_like(Tx)
    for i in range(len(Tx)):
        Ty[i] = lagrange(X, Y, Tx[i])
    return Tx,Ty

#wyznaczenie wielomianu Lagrange'a dla wezlow Czebyszewa
def calc_poly_lc(X,Y,a,b,n,is_rand=False):
    if is_rand:
        np.random.seed(1234)
        Tx=np.random.uniform(a,b,n)
    else:
        Tx = np.fromfunction(lambda i: np.cos(th(i,n)) , (n,), dtype=float)
    Ty = np.zeros_like(Tx)
    for i in range(len(Tx)):
        Ty[i] = lagrange(X, Y, Tx[i])
    return Tx,Ty

#wyznaczenie wielomianu dla kubicznych funkcji sklejanych
def calc_poly_cs(X,Y,a,b,n,is_rand=False):
    if is_rand:
        np.random.seed(1234)
        Tx=np.random.uniform(a,b,n)
    else:
        Tx = np.linspace(a,b,n)
    Ty = np.zeros_like(Tx)
    cs=CubicSpline(X,Y)
    Ty=cs(Tx)
    return Tx,Ty


def show(A,B,C,D,E,x=""):
    plt.clf()
    plt.figure(figsize=(15,6))
    plt.ylim([-0.25, 1.1])
    plt.plot(C[0],C[1],label="funkcja dokładna",linewidth=8,color="red")
    plt.plot(D[0],D[1], label="kubiczne funkcje składane - równoodległe węzły", linewidth=3 ,color="yellow")
    plt.plot(B[0],B[1],label="wielomiany Lagrange'a - równoodległe węzły",linewidth=2,color="green",)
    #plt.plot(A[0],A[1],'o',color="red")
    plt.plot(E[0],E[1],label="wielomiany Lagrange'a - węzły Czebyszewa",linewidth=2,color="blue")
    plt.legend(loc="upper left")
    plt.title(f"Wielomiany interpolacyjne - {x}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"lab4/pictures/Wielomiany interpolacyjne - {x}",dpi=400)
    plt.show()


X,Y=nodes(-1,1,13,f1)
Xc,Yc=nodes_cz(-1,1,13,f1)
Xlc,Ylc=calc_poly_lc(Xc,Yc,-1,1,130)
Xl,Yl=calc_poly_l(X,Y,-1,1,130)
Xcs,Ycs=calc_poly_cs(X,Y,-1,1,130)
Xex,Yex=nodes(-1,1,100,f1)
show((X,Y),(Xl,Yl),(Xex,Yex),(Xcs,Ycs),(Xlc,Ylc),"funkcja Rungego")

#wyznaczenie norm wektorów błędu
def err_vector(a,b,f,nodes,calc):
    res=[]
    for i in range(4,51):
        X,Y=nodes(a,b,i,f)
        Xl,Yl=calc(X,Y,a,b,500,True)
        res.append(np.linalg.norm(f(Xl)-Yl))
    return res
    
n=[i for i in range(4,51)]

v1=err_vector(-1,1,f1,nodes,calc_poly_l)
v2=err_vector(-1,1,f1,nodes,calc_poly_cs)
v3=err_vector(-1,1,f1,nodes_cz,calc_poly_lc)

plt.clf()
plt.figure(figsize=(12,6))
plt.semilogy(n,v2,"-o",linewidth=2,label="kubiczne funkcje składane - równoodległe węzły",markersize=5,color="orange")
plt.semilogy(n,v1,"-o",linewidth=2,label="wielomiany Lagrange'a - równoodległe węzły",markersize=5,color="green")
plt.semilogy(n,v3,"-o",linewidth=2,label="wielomiany Lagrange'a - węzły Czebyszewa",markersize=5,color="blue")
plt.legend(loc="upper left")
plt.title("Norma wektora błedów dla róznych metod interpolacji - funkcja Rungego")
plt.xlabel("liczba węzłów")
plt.ylabel("wartosc normy")
plt.savefig("lab4/pictures/Norma wektora błedów dla róznych metod interpolacji - funkcja Rungego (zoom)",dpi=400)
plt.show()

v4=err_vector(0,np.pi*2,f2,nodes,calc_poly_l)
v5=err_vector(0,np.pi*2,f2,nodes,calc_poly_cs)
v6=err_vector(0,np.pi*2,f2,nodes_cz_t,calc_poly_lc)

plt.clf()
plt.figure(figsize=(12,6))
plt.semilogy(n,v5,"-o",label="kubiczne funkcje składane - równoodległe węzły",markersize=5,color="orange")
plt.semilogy(n,v6,"-o",label="wielomiany Lagrange'a - węzły Czebyszewa",markersize=5,color="blue")
plt.semilogy(n,v4,"-o",label="wielomiany Lagrange'a - równoodległe węzły",markersize=5,color="green")
plt.legend(loc="upper right")
plt.title("Norma wektora błedów dla róznych metod interpolacji - funkcja 2")
plt.xlabel("liczba węzłów")
plt.ylabel("wartosc normy")
plt.savefig("lab4/pictures/Norma wektora błedów dla róznych metod interpolacji - funkcja 2",dpi=400)
plt.show()

