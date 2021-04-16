import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os
from matplotlib import animation
import time

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


def get_F(X, k,l0,C):
    disp, dist = get_disp(X)
    norm = disp / dist
    norm[np.isnan(norm)] = 0
    F = -(k * (dist - l0) * norm * C).sum(axis=1)
    return F

def simulate(X0,k,l0,C,d,dt,tfin):
    t_span = np.arange(0,tfin,dt)
    nt = t_span.size
    n_p = X0.shape[1]
    X_save = np.zeros((nt,2,n_p))
    X = X0.copy()
    for i, t in enumerate(t_span):
        F = get_F(X, k,l0,C)
        noise = np.sqrt(4*d*dt)*np.random.normal(0,1,F.shape) ##double check this is correct -- re. the "4"
        X += dt*F
        X += noise
        X_save[i] = X
    return X_save


def animate(X_save,C,n_frames=100, file_name=None, dir_name="plots",xlim=(0,10),ylim=(0,10)):

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    skip = int((X_save.shape[0]) / n_frames)

    def animate(i):
        ax1.cla()

        X = X_save[i*skip]
        ax1.scatter(X[0],X[1])

        C_u = np.triu(C,0)

        x1,x2 = np.meshgrid(X[0],X[0],indexing="ij")
        y1,y2 = np.meshgrid(X[1],X[1],indexing="ij")

        x1,x2,y1,y2 = x1[C_u],x2[C_u],y1[C_u],y2[C_u]

        ax1.plot(np.array((x1,x2)),np.array((y1,y2)))


        ax1.set(aspect=1, xlim=xlim, ylim=ylim)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)
    if file_name is None:
        file_name = "animation %d" % time.time()
    an = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
    an.save("%s/%s.mp4" % (dir_name, file_name), writer=writer, dpi=264)


N = 10
X0 = np.array((np.linspace(0,10,N),np.zeros(N)))

k = 0.1
l0 = 0.5
lmax = 1.5

C = np.zeros((N,N),dtype=np.bool_)##connectivity matrix
for i in range(N-1):
    C[i,i+1] = 1
for i in range(N-1):
    C[i+1,i] = 1


X_save = simulate(X0,k,l0,C,1e-4,0.001,100)

fig, ax = plt.subplots()
ax.plot(X_save[:,0],X_save[:,1])
fig.show()

animate(X_save,C,n_frames = 40,file_name="string",ylim=(-0.2,0.2))