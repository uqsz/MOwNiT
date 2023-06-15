import numpy as np
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,title
import matplotlib.pyplot as plt

def psi(x,y,L,n1,n2):
    return (2/L)*np.sin((n1*np.pi*(x+L/2)/L))*np.sin((n2*np))

# print(Z22)
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
from deepxde.backend import tf

L=2

def pde(x, psi):
    laplacian = dde.grad.hessian(psi, x, i=0, j=0) + dde.grad.hessian(psi, x, i=1, j=1)
    V = (1**2 + 1**2) * np.pi**2 / (2 * 2**2)
    return 1/2 * laplacian + V * psi

def solve(pde):
    geom = dde.geometry.Rectangle([-L / 2, -L / 2], [L / 2, L / 2])

    bc = dde.DirichletBC(geom, lambda _: -1, lambda x, on_boundary: on_boundary)

    data = dde.data.PDE(geom, pde, bc, num_domain=132, num_boundary=30, num_test=100)

    # Aktualizacja definicji sieci z uwzględnieniem parametrów n1 i n2
    net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)

    losshistory, train_state = model.train(iterations=2000)

    return model


def draw_model( model ):
    x = np.linspace(-L / 2, L / 2, 100)
    y = np.linspace(-L / 2, L / 2, 100)
    X, Y = np.meshgrid(x, y)
    xy = np.vstack([X.flatten(), Y.flatten()]).T
    pred = model.predict(xy)
    Z = pred.reshape(X.shape)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, Z, shading='auto')
    plt.colorbar()
    plt.show()
    
model1 = solve(pde)
draw_model(model1)