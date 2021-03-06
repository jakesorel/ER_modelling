import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os
from matplotlib import animation
import time
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


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


def get_F(dist,disp, k,l0,C):
    norm = disp / dist
    norm[np.isnan(norm)] = 0
    F = -(k * (dist - l0) * norm * C).sum(axis=1)
    return F

def get_C(dist,lmax=1.5):
    return (dist<lmax)*(~np.eye(dist.shape[0],dtype=np.bool_))


def simulate(X0,k,l0,lmax,d,dt,tfin):
    t_span = np.arange(0,tfin,dt)
    nt = t_span.size
    n_p = X0.shape[1]
    X_save = np.zeros((nt,2,n_p))
    X = X0.copy()
    for i, t in enumerate(t_span):
        disp, dist = get_disp(X)
        C = get_C(dist,lmax)
        F = get_F(dist,disp, k,l0,C)
        noise = np.sqrt(4*d*dt)*np.random.normal(0,1,F.shape) ##double check this is correct -- re. the "4"
        X += dt*F
        X += noise
        X_save[i] = X
    return X_save


def animate(X_save,lmax,n_frames=100, file_name=None, dir_name="plots",xlim=(0,10),ylim=(0,10)):

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    skip = int((X_save.shape[0]) / n_frames)

    def animate(i):
        ax1.cla()

        X = X_save[i*skip]
        disp, dist = get_disp(X)
        C = get_C(dist,lmax)
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


N = 60
# X0 = np.array((np.linspace(0,10,N),np.zeros(N)))
X0 = np.random.uniform(0,6,(2,N))
k = 0.1
l0 = 0.5
lmax = 1.5
dt,tfin = 0.05,100

C = np.zeros((N,N),dtype=np.bool_)##connectivity matrix
for i in range(N-1):
    C[i,i+1] = 1
for i in range(N-1):
    C[i+1,i] = 1


X_save = simulate(X0,k,l0,lmax,5e-2,dt,tfin)

fig, ax = plt.subplots()
ax.plot(X_save[:,0],X_save[:,1])
fig.show()

animate(X_save,lmax,n_frames = 25,file_name="string",xlim=(X_save[:,0].min(),X_save[:,0].max()),ylim=(X_save[:,1].min(),X_save[:,1].max()))


n_is = []
for i in np.arange(0,int(tfin/dt)):
    n_i = connected_components(csgraph=csr_matrix(get_C(get_disp(X_save[i])[1],lmax)), directed=False)[0]
    n_is.append(n_i)