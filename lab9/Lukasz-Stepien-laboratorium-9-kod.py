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
    


