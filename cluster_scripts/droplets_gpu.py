import gpe_fns_gpu as gpe
import mlx.core as cp
import numpy as np
import sys
import os
import plot_fns_gpu as plt_fns
import csv
import time
from matplotlib import pyplot as plt
import random

# physical constants
hbar = 1.05e-34 # J.s
a0 = 5.29e-11 # m

# read in simulation parameters
sim_name = "sys.argv[1]"
print("Simulation name: ", sim_name)
file_name = sim_name + ".csv"
# with open(file_name, mode='r') as file:
#     csvReader = csv.DictReader(file)
#     for line in csvReader:
#         args = line
# print("read sim params:", args)

# simulation parameters

Nx = 2**8
Ny = 2**8
Nz = 2**7
dt = 1e-2
Natoms = 2e2
Lx = 40
Ly = 40
Lz = 40

grid_shape = [Nx, Ny, Nz]
tolerance = 1e-4
iterations = int(1000)
ascats = [66*a0]
runs = 3

for ascat in ascats:
    # experimental parameters
    wx = 2*np.pi*57
    wy = 2*np.pi*390
    wz = 2*np.pi*98
    add = 66*a0
    m = 166*1.66e-27
    g = 4*np.pi*hbar**2*ascat/m
    epsdd = add/ascat
    gdd = 4*np.pi*hbar**2*add/m
    Q5edd = 1+3*epsdd**2/2
    Q5edd = 0
    
    gLHY = (32/(3*np.sqrt(np.pi)))*g*ascat**(3/2)*Q5edd
    
    # dimensionless units
    l = np.sqrt(hbar/(m*wx))
    tau = 1/wx
    
    print(l)
    
    gamma = Natoms*g/(hbar*wx*l**3)
    dipole_strength = gdd*Natoms/(hbar*wx*l**3)
    LHY_strength = gLHY*Natoms**(3/2)/(hbar*wx*l**(9/2))
    anisotropy = [1, (wy/wx)**2, (wz/wx)**2]
    dipole_direction = [0, 0, 1]
    
    
    # simulation parameters
    grid_shape = [Nx, Ny, Nz]
    time_step = dt
    grid_boundaries = [-Lx/2, Lx/2, -Ly/2, Ly/2, -Lz/2, Lz/2]
    
    potential_fn = lambda psi, grid, kgrid: gpe.dipolar_potential_LHY_cutoff(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction,LHY_strength)
    energy_fn = lambda psi, grid, kgrid: gpe.energy_functional_dipole_LHY(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction,LHY_strength)
    # guess = gpe.Gaussian_ansatz_three(grid_shape, grid_boundaries, 0.5, 0.55, 2.55, 4, 0.1)
    
    for r in range(runs):
        guess = gpe.Gaussian_ansatz(grid_shape, grid_boundaries, random.random()*Lx/2, random.random()*Ly/2, random.random()*Lz/2, 0.2)
        save_folder = "droplets/"+sim_name
        save_name = save_folder+"/Er_as="+str(round(ascat/a0,2))+"_run_"+str(r)
        start_time = time.time()
        grid, psi, max_iter_LHY, errs, mus, converged, Es = gpe.run_simulation_ite(grid_shape,grid_boundaries,potential_fn,time_step,iterations,guess,tolerance=tolerance,iter_check=100, energy_fn=energy_fn, save_name=save_name)
        end_time = time.time()
        print("elapsed time:", end_time-start_time, "s")
        print("converged:", converged)
        
        # plot result
        n_sim = cp.abs(psi)**2
        n_sim= n_sim.get()
        plt_fns.plot_density_unnormalized_int(grid,n_sim, sum_axes=("z",), save_name=str(round(ascat/a0,2))+"_run_"+str(r)+"xy_density.png", save_folder=save_folder, data_labels=("n",), units=(l*1e6, "um"))
        plt_fns.plot_density_unnormalized_int(grid,n_sim,sum_axes=("y",), save_name=str(round(ascat/a0,2))+"_run_"+str(r)+"xz_density.png", save_folder=save_folder, data_labels=("n",), units=(l*1e6, "um"))
        
        plt_fns.plot_density_unnormalized_int(grid,[n_sim],sum_axes=("y","z"), save_name=str(round(ascat/a0,2))+"_run_"+str(r)+"x_density.png", save_folder=save_folder, units=(l*1e6, "um"))
        plt_fns.plot_density_unnormalized_int(grid,[n_sim],sum_axes=("x","y"), save_name=str(round(ascat/a0,2))+"_run_"+str(r)+"z_density.png", save_folder=save_folder, units=(l*1e6, "um"))
        
        plt.clf()
        # plot convergence
        iters_LHY = np.arange(0,max_iter_LHY+1,1)
        plt.plot(iters_LHY, errs, label="LHY")
        plt.title("Error")
        plt.yscale('log')
        plt.legend()
        plt.savefig(save_name+"_err.png")
        
        plt.clf()
        plt.plot(iters_LHY, mus, label="LHY")
        plt.title("Chemical Potential")
        plt.legend()
        plt.savefig(save_name+"_mu.png")
        plt.clf()
        
        plt.close()
