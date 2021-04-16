import numpy as np
import matplotlib.pyplot as plt
from numba import jit

def plot_x(X):
    fig, ax = plt.subplots()
    ax.scatter(X[0],X[1])
    fig.show()


@jit(nopython=True)
def get_disp(X):
    x,y = X
    dx = np.outer(x,np.ones_like(x)) - np.outer(np.ones_like(x),x)
    dy = np.outer(y,np.ones_like(y)) - np.outer(np.ones_like(y),y)
    disp = np.dstack((dx,dy)).T
    dist = np.sqrt(dx**2 + dy**2)
    return disp, dist


def get_F(X, k,C):
    disp, dist = get_disp(X)
    norm = disp / dist
    norm[np.isnan(norm)] = 0
    F = -(k * (dist - l0) * norm * C).sum(axis=1)
    return F

def simulate(X0,k,C,dt,tfin):
    t_span = np.arange(0,tfin,dt)
    nt = t_span.size
    n_p = X0.shape[1]
    X_save = np.zeros((nt,2,n_p))
    X = X0.copy()
    for i, t in enumerate(t_span):
        F = get_F(X, k,C)
        X += dt*F
        X_save[i] = X
    return X_save

N = 10
X0 = np.array((np.linspace(0,10,N),np.zeros(N)))

k = 0.1
l0 = 0.5

C = np.zeros((N,N))##connectivity matrix
for i in range(N-1):
    C[i,i+1] = 1
for i in range(N-1):
    C[i+1,i] = 1


X_save = simulate(X0,k,C,0.001,100)

fig, ax = plt.subplots()
for i in range(10):
    ax.plot(np.arange(0,100,0.001),X_save[:,0,i])
ax.set(xlabel="t",ylabel="x")
fig.show()
