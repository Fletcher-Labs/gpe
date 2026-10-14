gpu = False
import numpy as np
from matplotlib import pyplot as plt
if gpu:
    import cupy as np

# functions
def run_simulation(grid_shape, grid_boundaries, potential, dt, iterations, guess, plot=False, iter_plot=0):
    grid,kgrid,psi = initialize_simulation(grid_shape, grid_boundaries, guess)
    xmin, xmax, ymin, ymax = np.ravel(grid_boundaries)
    for i in range(iterations):
        psi = propagation_step(psi,grid,kgrid,potential,dt)
        if plot:
            if i % iter_plot == 0:
                plt.imshow(np.abs(psi))
                plt.show()
                plt.imshow(np.angle(psi))
                plt.show()
    return grid, psi 

def initialize_simulation(grid_shape, grid_boundaries, guess):
    xmin, xmax, ymin, ymax = np.ravel(grid_boundaries)
    x = np.linspace(xmin,xmax,num=grid_shape[0])
    y =  np.linspace(ymin,ymax,num=grid_shape[1])
    Nx, Ny = grid_shape
    kx = 2*np.pi*np.fft.fftfreq(Nx)*Nx/(xmax-xmin)
    ky =  2*np.pi*np.fft.fftfreq(Ny)*Ny/(ymax-ymin)
    grid = np.meshgrid(x,y,sparse=True) 
    kgrid = np.meshgrid(kx,ky,sparse=True) 
    psi = guess
    plt.pcolormesh(x,y,np.abs(psi)**2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Initial Guess")
    plt.show()
    return (grid,kgrid,psi)

def propagation_step(psi, grid, kgrid, potential, dt):
    x, y = grid
    Nx, Ny = np.size(x), np.size(y)
    Lx, Ly = np.max(x)-np.min(x), np.max(y)-np.min(y)
    dLx, dLy = Lx/(Nx-1), Ly/(Ny-1)
    psi = kinetic_propagator(psi, grid, kgrid, potential, dt/2)
    psi = potential_propagator(psi, grid, kgrid, potential, dt)
    psi = kinetic_propagator(psi, grid, kgrid, potential, dt/2)
    psi_norm = np.sum(np.abs(psi)**2*dLx*dLy)
    return psi/np.sqrt(psi_norm)

def harmonic_potential(psi,grid,gamma,anisotropy):
    x, y = grid
    return 1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2)+gamma*np.abs(psi)**2

def potential_propagator(psi, grid, kgrid, potential, dt):
    return np.exp(-1*potential(grid,psi)*dt)*psi

def kinetic_propagator(psi, grid, kgrid, potential, dt):
    fft_psi = np.fft.fftn(psi)
    kx, ky = kgrid
    ksquared = kx**2+ky**2 
    return np.fft.ifftn(np.exp(-1/2*ksquared*dt)*fft_psi)

def Gaussian_ansatz(grid_shape, grid_boundaries, anisotropy):
    xmin, xmax, ymin, ymax = np.ravel(grid_boundaries)
    x = np.linspace(xmin,xmax,num=grid_shape[0])
    y =  np.linspace(ymin,ymax,num=grid_shape[1])
    grid = np.meshgrid(x,y,sparse=True) 
    x, y = grid
    Nx, Ny = np.size(x), np.size(y)
    Lx, Ly = np.max(x)-np.min(x), np.max(y)-np.min(y)
    psi = np.exp(-(np.sqrt(anisotropy[0])*x**2+np.sqrt(anisotropy[1])*y**2)/2)
    dLx, dLy = Lx/(Nx-1), Ly/(Ny-1)
    psi_norm = np.sum(np.abs(psi)**2*dLx*dLy)
    return psi/np.sqrt(psi_norm)
