from numpy import arange,sin,pi
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import latexify
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def psi(x,y,L,n1,n2):
    return (2/L)*sin((n1*pi*(x+L/2)/L))*sin((n2*pi*(y+L/2)/L))


def exact_graph(n1,n2):
    plt.clf()
    x = arange(-1.0,1.0,10**(-2))
    y = arange(-1.0,1.0,10**(-2))
    X,Y = meshgrid(x, y) 
    Z = psi(X,Y,2,n1,n2) # evaluation of the function on the grid
    im = imshow(Z,cmap=cm.coolwarm) # drawing the function
    cset = contour(Z,arange(-1,1.5,0.2),linewidths=1)
    clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
    colorbar(im) 
    title(r'$\psi(x, y) = sin{({\frac{n_1 {\pi} \left( x + 1\right)}{2}})} \sin{\left({\frac{n_2 {\pi} \left( y + 1 \right)}{2}} \right)},  (n_1,n_2)=$'+f'({n1},{n2})')
    plt.savefig(f"lab10/pictures/Wykres konturowy funkcji ψ(x,y) dla (n1,n2)=({n1},{n2})", dpi=350)

    # plt.show()
    
exact_graph(1,1)
exact_graph(1,2)
exact_graph(2,1)
exact_graph(2,2)