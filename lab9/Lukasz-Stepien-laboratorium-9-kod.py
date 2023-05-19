from matplotlib import axes
import matplotlib.pyplot as plt
import numpy as np
# import os

# os.chdir('C:/Users/user/Desktop/AGH/mownit')

# zadanie 2

def f(y,t):
    return 5*y

def euler(t_end,h,f):
    Tx=[0]
    Ty=[1]
    t=h
    i=1
    while t<t_end:
        Ty.append(Ty[i-1]-f(Ty[i-1],t)*h)
        Tx.append(t)
        i+=1
        t+=h
    return Tx, Ty

Tx, Ty = euler(10,0.5,f)

plt.clf()
plt.plot(Tx, Ty, '-o')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Metoda Eulera dla kroku h=0.5')
plt.grid(True)
plt.savefig("lab9/pictures/Metoda Eulera dla kroku h=0,5", dpi=350)
# plt.show()

def niejawny_euler(t_end,h,f):
    Tx=[0]
    Ty=[1]
    t=h
    i=1
    while t<t_end:
        Ty.append(Ty[i-1]*(1/(1+5*h)))
        Tx.append(t)
        i+=1
        t+=h
    return Tx, Ty

Tx, Ty = niejawny_euler(10,0.5,f)

plt.clf()
plt.plot(Tx, Ty, '-o')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Niejawna metoda Eulera dla kroku h=0.5')
plt.grid(True)
plt.savefig("lab9/pictures/Niejawna metoda Eulera dla kroku h=0,5", dpi=350)
# plt.show()


#zadanie 3a

def r(x,y):
    return np.sqrt(x**2+y**2)

def pze(xp,yp,x,y):
    return 0.5*(xp**2+yp**2)-1/r(x,y)

def pzmp(xp,yp,x,y):
    return x*yp-y*xp

def show(Tvx,Tx,Tvy,Ty,Tt,title):
    plt.clf()
    plt.plot(Tt,Ty)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Wykres y(t) -> '+title)
    plt.grid(True)
    plt.savefig("lab9/pictures/"+'Wykres y(t) '+title, dpi=350)
    # plt.show()
    
    plt.clf()
    plt.plot(Tt,Tx)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Wykres x(t) -> '+title)
    plt.grid(True)
    plt.savefig("lab9/pictures/"+'Wykres x(t) '+title, dpi=350)
    # plt.show()
    
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(Ty,Tx)
    plt.xlabel('y')
    plt.ylabel('x')
    plt.title('Wykres x(y) -> '+title)
    plt.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.savefig("lab9/pictures/"+'Wykres x(y) '+title, dpi=350)
    # plt.show()
    
    Tr=[r(Tx[i],Ty[i]) for i in range(len(Tx))]
    Tvr=[r(Tvx[i],Tvy[i]) for i in range(len(Tx))]
    
    plt.clf()
    plt.plot(Tr,Tvr, label='y1')
    plt.xlabel('r')
    plt.ylabel('vr')
    plt.title('Wykres vr(r) -> '+title)
    plt.grid(True)
    plt.savefig("lab9/pictures/"+'Wykres vr(r) '+title, dpi=350)
    # plt.show()
    
    Tpze=[pze(Tvx[i],Tvy[i],Tx[i],Ty[i]) for i in range(len(Tx))]
    
    plt.clf()
    plt.plot(Tt,Tpze, label='y1')
    plt.xlabel('t')
    plt.ylabel('E')
    plt.title('Wykres energii E(t) -> '+title)
    plt.grid(True)
    plt.savefig("lab9/pictures/"+'Wykres energii E(t) '+title, dpi=350)
    # plt.show()
    
    Tpzmp=[pzmp(Tvx[i],Tvy[i],Tx[i],Ty[i]) for i in range(len(Tx))]
    plt.clf()
    plt.plot(Tt,Tpzmp, label='y1')
    plt.xlabel('t')
    plt.ylabel('p')
    plt.title('Wykres momentu pedu p(t) -> '+title)
    plt.grid(True)
    plt.savefig("lab9/pictures/"+'Wykres momentu pedu p(t) '+title, dpi=350)
    # plt.show()


def f(x,y,t):
    return (x**2+y**2)**(-1.5)

def euler_mul(t_end,h,e):
    Tx=[1-e]
    Ty=[0]
    Tvx=[0]
    Tvy=[np.sqrt((1+e)/(1-e))]
    Tt=[0]
    t=0
    i=0
    while t<t_end:
        Tvx.append(Tvx[i]-Tx[i]*f(Tx[i],Ty[i],t)*h)
        Tx.append(Tx[i]+Tvx[i]*h)
        Tvy.append(Tvy[i]-Ty[i]*f(Tx[i],Ty[i],t)*h)
        Ty.append(Ty[i]+Tvy[i]*h)
        Tt.append(t)
        i+=1
        t+=h
    return Tvx,Tx,Tvy,Ty,Tt
        
Tvx,Tx,Tvy,Ty,Tt=euler_mul(40,10**(-3),0)
show(Tvx,Tx,Tvy,Ty,Tt,"jawna metoda Eulera")

#zadanie 3b

def f(x1, x2):
    return (x1**2 + x2**2)**(-3/2)

def jacobian(x,h):
    epsilon=1e-12
    x1, x2, x3, x4 = x
    
    df_dx1 = (f(x1 + epsilon, x2) - f(x1 - epsilon, x2)) / (2 * epsilon)
    df_dx2 = (f(x1, x2 + epsilon) - f(x1, x2 - epsilon)) / (2 * epsilon)

    jacobian = np.zeros((4, 4))
    jacobian[0, 0] = 1
    jacobian[0, 3] = -h
    jacobian[1, 1] = 1
    jacobian[1, 2] = -h
    jacobian[2, 0] = h * x2 * df_dx1
    jacobian[2, 1] = h*(f(x1,x2) + x2 * df_dx2)
    jacobian[2, 2] = 1
    jacobian[3, 0] = h*(f(x1,x2) + x1 * df_dx1)
    jacobian[3, 1] = h * x1 * df_dx2
    jacobian[3, 3] = 1

    return jacobian

def evaluate_equations(x,a,b,c,d,h):
    x1, x2, x3, x4 = x
    equations = np.zeros(4)
    equations[0] = x1 - a - x4 * h
    equations[1] = x2 - b - x3 * h
    equations[2] = x3 - c + x2 * f(x1, x2) * h
    equations[3] = x4 - d + x1 * f(x1, x2) * h
    return equations

def solve_equations(a, b, c, d, h, tol=1e-12, max_iter=100):
    x = np.array([1,1,1,1], dtype=float)
    i=0
    while True:
        x_prev = x.copy()
        delta_x = np.linalg.solve(jacobian(x_prev,h), -evaluate_equations(x_prev,a, b, c, d, h))
        x += delta_x
        i+=1
        if np.linalg.norm(delta_x) < tol or i>max_iter:
            return x
    

def euler_mul1(t_end,h,e):
    Tx=[1-e]
    Ty=[0]
    Tvx=[0]
    Tvy=[np.sqrt((1+e)/(1-e))]
    Tt=[0]
    t=0
    i=0
    while t<t_end:
        x1,x2,x3,x4 = solve_equations(Ty[i], Tx[i], Tvx[i], Tvy[i], h) 
        Ty.append(x1)
        Tx.append(x2)
        Tvx.append(x3)
        Tvy.append(x4)
        Tt.append(t)
        i+=1
        t+=h
    return Tvx,Tx,Tvy,Ty,Tt

Tvx,Tx,Tvy,Ty,Tt=euler_mul1(40,10**(-3),0)
show(Tvx,Tx,Tvy,Ty,Tt,"niejawna metoda Eulera")

#zadanie 3c

def fx(x,y):
    return -x*(x**2+y**2)**(-3/2)

def fy(x,y):
    return -y*(x**2+y**2)**(-3/2)

def euler_poljawny(t_end,h,e):
    Tx=[1-e]
    Ty=[0]
    Tvx=[0]
    Tvy=[np.sqrt((1+e)/(1-e))]
    Tt=[0]
    t=0
    i=0
    while t<t_end:
        Tvx.append(Tvx[i]+h*fx(Tx[i],Ty[i]))
        Tx.append(Tx[i]+h*Tvx[i+1])
        Tvy.append(Tvy[i]+h*fy(Tx[i],Ty[i]))
        Ty.append(Ty[i]+h*Tvy[i+1])
        Tt.append(t)
        i+=1
        t+=h
    return Tvx,Tx,Tvy,Ty,Tt

Tvx,Tx,Tvy,Ty,Tt=euler_poljawny(40,10**(-3),0)
show(Tvx,Tx,Tvy,Ty,Tt,"poljawna metoda Eulera")

#zadanie 3d

def runge_kutta(t_end,h,e):
    Tx=[1-e]
    Ty=[0]
    Tvx=[0]
    Tvy=[np.sqrt((1+e)/(1-e))]
    Tt=[0]
    t=0
    i=0
    while t<t_end:
        k1=Tvx[i]
        k2=Tvx[i]+h*k1*0.5
        k3=Tvx[i]+h*k2*0.5
        k4=Tvx[i]+h*k3
        Tx.append(Tx[i]+(h/6)*(k1+2*k2+2*k3+k4))
        
        k1=Tvy[i]
        k2=Tvy[i]+h*k1*0.5
        k3=Tvy[i]+h*k2*0.5
        k4=Tvy[i]+h*k3
        Ty.append(Ty[i]+(h/6)*(k1+2*k2+2*k3+k4))
        
        k1=fx(Tx[i],Ty[i])
        k2=fx(Tx[i]+h*k1*0.5,Ty[i]+h*k1*0.5)
        k3=fx(Tx[i]+h*k2*0.5,Ty[i]+h*k2*0.5)
        k4=fx(Tx[i]+h*k3,Ty[i]+h*k3)
        Tvx.append(Tvx[i]+(h/6)*(k1+2*k2+2*k3+k4))
        
        k1=fy(Tx[i],Ty[i])
        k2=fy(Tx[i]+h*k1*0.5,Ty[i]+h*k1*0.5)
        k3=fy(Tx[i]+h*k2*0.5,Ty[i]+h*k2*0.5)
        k4=fy(Tx[i]+h*k3,Ty[i]+h*k3)
        Tvy.append(Tvy[i]+(h/6)*(k1+2*k2+2*k3+k4))
        
        Tt.append(t)
        
        i+=1
        t+=h
    return Tvx,Tx,Tvy,Ty,Tt
    

Tvx,Tx,Tvy,Ty,Tt=runge_kutta(40,10**(-3),0)
show(Tvx,Tx,Tvy,Ty,Tt,"RK4")





































    


