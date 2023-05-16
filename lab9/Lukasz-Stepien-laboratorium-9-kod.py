import matplotlib.pyplot as plt

# zadanie 2

def f(y,t):
    return 5*y

def euler(a,b,h,f):
    Tx=[0]
    Ty=[1]
    t=h
    i=1
    while t<b:
        Ty.append(Ty[i-1]-f(Ty[i-1],t)*h)
        Tx.append(t)
        i+=1
        t+=h
    return Tx, Ty

Tx, Ty = euler(0,10,0.5,f)

plt.plot(Tx, Ty, '-o')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Metoda Eulera dla kroku h=0.5')
plt.grid(True)
plt.savefig(f"pictures/Metoda Eulera dla kroku h=0,5", dpi=350)
plt.show()

def niejawny_euler(a,b,h,f):
    Tx=[0]
    Ty=[1]
    t=h
    i=1
    while t<b:
        Ty.append(Ty[i-1]*(1/(1+5*h)))
        Tx.append(t)
        i+=1
        t+=h
    return Tx, Ty

Tx, Ty = niejawny_euler(0,10,0.5,f)

plt.plot(Tx, Ty, '-o')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Niejawna metoda Eulera dla kroku h=0.5')
plt.grid(True)
plt.savefig(f"pictures/Niejawna metoda Eulera dla kroku h=0,5", dpi=350)
plt.show()

def f(x,y,t):
    return (x**2+y**2)**(-1.5)

def euler_mul(a,b,n):
    Tx=[1]
    Ty=[0]
    Tvx=[0]
    Tvy=[1]
    Tt=[0]
    h=(b-a)/n
    t=0
    i=0
    while t<b:
        Tvx.append(Tvx[i]-Tx[i]*f(Tx[i],Ty[i],t)*h)
        Tx.append(Tx[i]+Tvx[i]*h)
        Tvy.append(Tvy[i]-Ty[i]*f(Tx[i],Ty[i],t)*h)
        Ty.append(Ty[i]+Tvy[i]*h)
        Tt.append(t)
        i+=1
        t+=h
    return Tvx,Tx,Tvy,Ty,Tt
        
Tvx,Tx,Tvy,Ty,Tt=euler_mul(0,100,20000)

plt.plot(Ty,Tx, label='y1')
plt.xlabel('Czas')
plt.ylabel('y1')
plt.title('Wykres y1')
plt.legend()
plt.grid(True)
plt.show()
    


