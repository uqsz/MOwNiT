import numpy as np
import matplotlib.pyplot as plt

years=[1900+i*10 for i in range(9)]
population=[76212168, 92228496, 106021537, 123202624, 132164569,
            151325798, 179323175, 203302031, 226542199]


# funkcje bazowe
def f1(t,j): 
    return t**j

def f2(t,j):
    return (t-1900)**j
    
def f3(t,j):
    return (t-1940)**j

def f4(t,j):
    return ((t-1940)/40)**j


# tworzenie macierzy Vandermonde'a
def vand(f,Y):
    return np.fromfunction(lambda i,j: f(Y[i.astype(int)],j) , (9,9), dtype=float)

# wspolczynniki uwarunkowania
def check_cond(F,Y):
    t=[]
    for i in range(4):
        temp=np.linalg.cond(vand(F[i],Y))
        t.append((temp,i+1))
        print(f"Współczynnik uwarunkowania dla V{i+1}:","{:e}".format(temp))
    m=min(t)
    print(f"\nNajmniejszy dla zbioru bazowego nr {m[1]}:", "{:e}".format(m[0]))

# algorytm hornera do wyliczania wartosci wielomianu
def horner(A, x, f):
    n=len(A)
    A=A[::-1]
    result = A[0] 
    for i in range(1, n):
        result = result*f(x,1) + A[i]
    return result

# wyliczanie wartosci wielomianu
def calc_poly_v(B,Y,f):
    X=vand(f,Y)
    A=np.linalg.solve(X, B)
    Tx = np.arange(1900, 1991)
    Ty=[]
    for i in range(1900,1991):
        if i%10==0 and i!=1990:
            Ty.append(B[(i//10)%10])
        else:
            Ty.append(horner(A,i,f))
    return Tx,Ty


# wyliczanie wartosci wielomianu interpolacyjnego Lagrange'a
def lagrange(Y, B, x):
    n = len(Y)
    L = np.zeros(n)
    for i in range(n):
        L[i] = np.prod((x - Y[np.arange(n)!=i])/(Y[i]-Y[np.arange(n)!=i]))
    return np.sum(B*L)

# wyznaczanie wartości wielomianu
def calc_poly_l(Y,B):
    Tx = np.arange(1900, 1981)
    Ty = np.zeros_like(Tx)
    for i in range(len(Tx)):
        Ty[i] = lagrange(Y, B, Tx[i])
    return Tx,Ty
    
def iloraz_roznicowy(x, y):
    n = len(x)
    ilorazy = np.zeros((n, n))
    ilorazy[:,0] = y
    for j in range(1, n):
        for i in range(n-j):
            ilorazy[i][j] = (ilorazy[i+1][j-1] - ilorazy[i][j-1]) / (x[i+j] - x[i])
    return ilorazy[0]

def interpolacja_newtona(x, y, xs):
    a = iloraz_roznicowy(x, y)
    n = len(x)
    res = np.zeros_like(xs)
    for i in range(n):
        temp = a[i]
        for j in range(i):
            temp *= (xs - x[j])
        res += temp
    return res

def calc_poly_n(Y,B):
    xs = np.linspace(1900, 1981, 80)
    ys = interpolacja_newtona(Y, B, xs)
    return xs,ys

# tworzenie wykresu
def show(T,x="",c="blue"):
    plt.clf()
    plt.title(f"Wielomian interpolacyjny - {x}")
    plt.xlabel("Lata")
    plt.xticks(years+[1990])
    plt.ylabel("Wielkosc populacji")
    plt.plot(T[0],T[1],color=c)
    plt.plot(years,population,'o',color="red")
    plt.savefig(f"lab3/pictures/Wielomian interpolacyjny - {x}")
    #plt.show()


def main():
    F=[f1,f2,f3,f4]

    B=np.array(population)
    Y=np.array(years)

    L_1990=248709873
    
    check_cond(F,Y)
    
    X=vand(f4,Y)
    A1=np.linalg.solve(X, B)
    
    x=calc_poly_v(B,Y,f4)
    print("\nBłąd względny dla roku 1990:", "{:.2%}".format((abs(x[1][90]-L_1990)/L_1990)))
    show(x,"Vandermonde")
    print("\nWspółczynniki wielomianu interpolacyjnego:")
    print(A1)
    x=calc_poly_l(Y,B)
    show(x,"Lagrange",c="green")
    
    x=calc_poly_n(Y,B)
    show(x,"Newton",c="violet")
    
    B=np.around(B,decimals=-6)

    x=calc_poly_v(B,Y,f4)
    show(x,"Vandermond (zaokrąglenie)","orange")
    
    A2=np.linalg.solve(X, B)
    A=abs((A1-A2)/A2)
    
    print("\nWzględne różnice pomiędzy współczynnika wielomianu z i bez zaokraglenia:")
    print([ "{:0.2%}".format(x) for x in A ])


main()

















