gpu = False
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import gpe_mixture as gpe
import analysis_fns as plt_fns
import time
import os

if gpu:
    import cupy as np
# import pyfftw.interfaces as fftw

# physical constants
hbar = 1.05e-34 # J.s
a0 = 5.29e-11 # m

def run_sim(a12, save_folder):

    # experimental parameters
    a1 = 50*a0
    a2 = 50*a0
    N1 = 1e4
    N2 = 1e4
    m1 = 7*1.66e-27
    m2 = 166*1.66e-27
    m12 = m1*m2/(m1+m2)
    g11 = 4*np.pi*hbar**2*a1/m1
    g22 = 4*np.pi*hbar**2*a2/m2
    g12 = 2*np.pi*hbar**2*a12/m12
    gratio = g11*g22/g12**2
    prediction = "miscible" if gratio > 1 else "immiscible"
    print(g11*g22/g12**2)
    print(prediction)
    

    # dimensionless units
    l = a1
    tau = m1*a1**2/hbar


    # scaled parameters
    mass_ratio = m2/m1
    gamma11 = N1*g11*m1/(a1*hbar**2)
    gamma12 =  N2*g12*m1/(a1*hbar**2)
    gamma21 = N1*g12*m1/(a1*hbar**2)
    gamma22 =  N2*g22*m1/(a1*hbar**2)
    gammas =  [gamma11, gamma12, gamma21, gamma22]
    print(gammas)
    print(gamma11*gamma22/(gamma12*gamma21))

    # # simulation grid
    Nx = 2**6
    Ny = 2**6
    Nz = 2**6
    grid_shape = [Nx, Ny, Nz]

    iterations = 5000
    tolerance = 1e-6

    time_step = 1e-3

    Lx = 8
    Ly = 8
    Lz = 8
    grid_boundaries = [-Lx/2, Lx/2, -Ly/2, Ly/2, -Lz/2, Lz/2]

    pad_fraction = 0.3
    potential_propagator = lambda psi, grid, kgrid, dt: gpe.box_potential_propagator(psi, grid, kgrid, dt, gammas, pad_fraction)
    kinetic_propagator = lambda psi, grid, kgrid, dt: gpe.box_kinetic_propagator(psi, grid, kgrid, dt, mass_ratio, pad_fraction)
    guess1 = gpe.Gaussian_ansatz(grid_shape, grid_boundaries, 5, 5, 5, noise_level=0.1)
    guess2 = gpe.Gaussian_ansatz(grid_shape, grid_boundaries, 5, 5, 5, noise_level=0.1)
    start_time = time.time()
    grid, psi, max_iter, errs, mu1s, mu2s, Es, converged = gpe.run_simulation_ite(grid_shape,grid_boundaries,potential_propagator, kinetic_propagator,time_step,iterations,guess1, guess2, tolerance=tolerance,iter_check=100, use_tolerance=True)
    end_time = time.time()
    print("elapsed time:", end_time-start_time, "s")
    print("converged:", converged)
    psi1 = psi[0,:,:,:]
    psi2 = psi[1,:,:,:]
    n1 = np.abs(psi1)**2
    n2 = np.abs(psi2)**2
    overlap = np.sum(n1*n2)/np.sqrt((np.sum(n1**2)*np.sum(n2**2)))
    print("overlap: ", overlap)

    plt_fns.plot_density_unnormalized_int(grid,[np.abs(psi[0,:,:,:])**2,np.abs(psi[1,:,:,:])**2], data_labels=("n1", "n2"), save_name="_"+str(round(gratio,2))+".png" , save_folder=save_folder, show=False)
    plt.close()

    return gratio, overlap

save_folder = "results/misc_box7/"
a12s = np.linspace(10*a0,30*a0,10)
overlaps = []
gratios = []
for a12 in a12s:
    gratio, overlap = run_sim(a12, save_folder)
    gratios.append(gratio)
    overlaps.append(overlap)
    
plt.scatter(gratios, overlaps)
plt.xlabel("g ratio")
plt.ylabel("Overlap integral")
plt.savefig(os.path.join(save_folder, "overlaps.png"))
np.savetxt(os.path.join(save_folder, "gratios.csv"), gratios, delimiter=',')
np.savetxt(os.path.join(save_folder, "overlaps.csv"), overlaps, delimiter=',')
plt.close()