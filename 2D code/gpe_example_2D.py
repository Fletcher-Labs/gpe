gpu = False
import numpy as np
from matplotlib import pyplot as plt
import gpe_functions_2D as gpe
if gpu:
    import cupy as np

# physical constants
hbar = 1.05e-34 # J.s
a0 = 5.29e-11 # m

# experimental parameters
wx = 2*np.pi*5e3 # Hz
wy = 2*np.pi*5e3*2
ascat = 30*a0
Natoms = 1e4
m = 7*1.66e-27
g = 4*np.pi*hbar**2*ascat/m

# dimensionless units
l = np.sqrt(hbar/(m*wx))
tau = 1/wx
gamma = Natoms*g/(hbar*wx*l**3)
anisotropy = [1,(wy/wx)**2]

# simulation grid
xmin = -10
xmax = 10
ymin = -10
ymax = 10
Nx = 2**7
Ny = 2**7
grid_shape = [Nx, Ny]
grid_boundaries = [[xmin,xmax],[ymin,ymax]]

# simulation parameters
time_step = 1e-3
iterations = 1000 # replace with a smarter convergence parameter

# set up potential and initial guess
potential = lambda grid, psi: gpe.harmonic_potential(psi, grid, gamma, anisotropy)
guess = gpe.Gaussian_ansatz(grid_shape, grid_boundaries, anisotropy)
grid, psi = gpe.run_simulation(grid_shape,grid_boundaries,potential,time_step,iterations,guess,plot=False,iter_plot=0)

x,y=grid
Nx, Ny = np.size(x), np.size(y)
Lx, Ly = np.max(x)-np.min(x), np.max(y)-np.min(y)

n_sim = np.abs(psi)**2
plt.pcolormesh(x,y,n_sim)
plt.xlabel("x")
plt.ylabel("y")
plt.title("GPE Density")
plt.show()

def thomas_fermi_density(grid,gamma,anisotropy):
    x,y = grid
    Nx, Ny = np.size(x), np.size(y)
    Lx, Ly = np.max(x)-np.min(x), np.max(y)-np.min(y)
    dLx, dLy = Lx/Nx, Ly/Ny
    potential = 1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2)
    mu = np.sqrt(gamma/np.pi*np.sqrt(anisotropy[0]*anisotropy[1]))
    tf = (mu-potential)/gamma
    tf[tf<0] = 0
    tf_norm = np.sum(tf*dLx*dLy)
    return tf/tf_norm


tf_n = thomas_fermi_density(grid,gamma,anisotropy)
plt.pcolormesh(x,y,tf_n)
plt.xlabel("x")
plt.ylabel("y")
plt.title("TF Profile")
plt.show()

n_int_sim = np.ravel(np.sum(n_sim, axis=0)/Ny)
tf_int = np.ravel(np.sum(tf_n, axis=0)/Ny)
plt.plot(np.ravel(x),n_int_sim,label="Simulation profile")
plt.plot(np.ravel(x),tf_int,label="Thomas-Fermi profile")
plt.xlabel("x")
plt.ylabel("nx")
plt.legend()
plt.title("1D Density")
plt.show()

n_int_sim = np.ravel(np.sum(n_sim, axis=1)/Nx)
tf_int = np.ravel(np.sum(tf_n, axis=1)/Nx)
plt.plot(np.ravel(y),n_int_sim,label="Simulation profile")
plt.plot(np.ravel(y),tf_int,label="Thomas-Fermi profile")
plt.xlabel("y")
plt.ylabel("ny")
plt.legend()
plt.title("1D Density")
plt.show()
