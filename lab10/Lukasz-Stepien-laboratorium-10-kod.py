import numpy as np
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,title
import matplotlib.pyplot as plt

def psi(x,y,L,n1,n2):
    return (2/L)*np.sin((n1*np.pi*(x+L/2)/L))*np.sin((n2*np.pi*(y+L/2)/L))

def exact_graph(n1,n2):
    plt.clf()
    if n2==1: tmp=1 
    else: tmp=-1
    x = np.arange(-1.0,1.0,10**(-2))
    y = np.arange(-1.0,1.0,10**(-2))
    X,Y = meshgrid(x, y) 
    Z = psi(X,Y,2,n1,n2)
    Z[:, 0] = 0
    Z[:, -1] = 0
    Z[0, :] = 0
    Z[-1, :] = 0
    im=imshow(Z,cmap=cm.hot, extent=[-1, 1, -1, 1])
    cset = contour(x,y,tmp*Z, levels=np.arange(-1,1, 0.1),linewidths=1,extend='both')
    clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    colorbar(im)
    title(r'$\psi(x, y) = sin{({\frac{n_1 {\pi} \left( x + 1\right)}{2}})} \sin{\left({\frac{n_2 {\pi} \left( y + 1 \right)}{2}} \right)},  (n_1,n_2)=$'+f'({n1},{n2})')
    # plt.savefig(f"lab10/pictures/Wykres konturowy funkcji ψ(x,y) dla (n1,n2)=({n1},{n2})", dpi=350)
    plt.show()
    return Z
    
Z11=exact_graph(1,1)
Z12=exact_graph(1,2)
Z21=exact_graph(2,1)
Z22=exact_graph(2,2)

# print(Z22)