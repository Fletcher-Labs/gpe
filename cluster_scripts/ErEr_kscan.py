from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
from cupy import fft as fftw
import os
import numpy
import cupy as np
import sys
import csv
import time
from gpe_mixture_fns_gpu import *
import random

# read in simulation parameters
sim_name = sys.argv[1]
save_folder = "results/"+sim_name
print("Simulation name: ", sim_name)
# physical constants
hbar = 1.05e-34 # J.s
a0 = 5.29e-11 # m
# inputs
tolerance = 1e-7
iterations = 100000
time_step = 1e-1
noise_iters = 0
ani = 1000

def setup_sim():
    # experimental parameters
    wx = 2*np.pi*10
    wy = 2*np.pi*150
    wz = 2*np.pi*150
    a1 =  65*a0
    a12 = 60*a0
    a2 = 65*a0
    add = 64*a0
    N1 = 2e4
    N2 = 2e4
    m1 = 166*1.66e-27
    m2 = 166*1.66e-27
    m12 = m1*m2/(m1+m2)
    g11 = 4*np.pi*hbar**2*a1/m1
    g22 = 4*np.pi*hbar**2*a2/m2
    g12 = 2*np.pi*hbar**2*a12/m12
    gdd = 4*np.pi*hbar**2*add/m1
    pol_ratio = 1
    dipole_direction = [0, 0, 1]

    gratio = g11*g22/g12**2
    prediction = "miscible" if gratio > 1 else "immiscible"
    print(g11*g22/g12**2)
    print(prediction)

    # dimensionless units
    l = np.sqrt(hbar/(m1*wx))
    tau = 1/wx
    print(l)

    # scaled parameters
    atom_nums = [N1, N2]
    anisotropy = [1, (wy/wx)**2, (wz/wx)**2]
    mass_ratio = m2/m1
    gamma11 = N1*g11/(hbar*wx*l**3)
    gamma12 =  N2*g12/(hbar*wx*l**3)
    gamma21 = N1*g12/(hbar*wx*l**3)
    gamma22 =  N2*g22/(hbar*wx*l**3)
    gammas =  [gamma11, gamma12, gamma21, gamma22]
    print(gammas)
    print(gamma11*gamma22/(gamma12*gamma21))
    print("gdd:",gdd)
    d1 = gdd*N1/(hbar*wx*l**3)
    d2 = gdd*N2/(hbar*wx*l**3)
    d12 = gdd*N2/(hbar*wx*l**3)
    d21 = gdd*N1/(hbar*wx*l**3)
    dipole_strengths = [d1, d12*0, d21*0, d2*0]
    edd1 = d1/gamma11
    edd2 = d2/gamma22

    Q5edd1 = 1+3*edd1**2/2
    Q5edd2 = 1+3*edd2**2/2

    # y = (1-epsdd)/(3*epsdd)+0j
    # Q5edd = (3*epsdd)**(5/2)/48*((8+26*y+33*y**2)*np.sqrt(1+y) + 15*y**3*np.log((1+np.sqrt(1+y))/np.sqrt(y)))
    # assert (np.real(Q5edd) > 10*np.imag(Q5edd))
    # Q5edd = np.real(Q5edd)
    # print(Q5edd)

    gLHY1 = (32/(3*np.sqrt(np.pi)))*g11*a1**(3/2)*Q5edd1
    gLHY2 = (32/(3*np.sqrt(np.pi)))*g22*a2**(3/2)*Q5edd2

    LHY_factor1 = gLHY1*N1**(3/2)/(hbar*wx*l**(9/2))
    LHY_factor2 = gLHY2*N2**(3/2)/(hbar*wx*l**(9/2))
    LHY_factors = [LHY_factor1, LHY_factor2]

    # # simulation grid
    gamma = max(gammas)
    mu = (15*gamma/(8*2**(3/2)*np.pi)*np.sqrt(anisotropy[0]*anisotropy[1]*anisotropy[2]))**(2/5)
    RTF = [np.sqrt(2*mu/anisotropy[0]), np.sqrt(2*mu/anisotropy[1]), np.sqrt(2*mu/anisotropy[2])]
    Lx = 4*RTF[0]
    Ly = 4*RTF[1]
    Lz = 4*RTF[2]
    print("Ls:", Lx, Ly, Lz)

    Nx = 2**9
    Ny = 2**7
    Nz = 2**7

    grid_shape = [Nx, Ny, Nz]
    grid_boundaries = [-Lx/2, Lx/2, -Ly/2, Ly/2, -Lz/2, Lz/2]

    potential_propagator = lambda psi, grid, kgrid, dt: double_dipolar_potential_cutoff_intra_LHY_propagator(psi, grid, kgrid, dt, gammas, anisotropy, pol_ratio, dipole_strengths, dipole_direction, LHY_factors, sag=0)
    kinetic_propagator = lambda psi, grid, kgrid, dt: mixture_kinetic_propagator(psi, grid, kgrid, dt, mass_ratio)
    energy_fn = lambda psi, grid, kgrid: energy_functional_double_dipolar_cutoff_intra_LHY(psi,grid,kgrid,gammas,anisotropy, pol_ratio, mass_ratio, dipole_strengths, dipole_direction, atom_nums, LHY_factors, sag=0)
    return grid_shape, grid_boundaries, potential_propagator, kinetic_propagator, energy_fn, l

def run_sim(kx):
    grid_shape, grid_boundaries, potential_propagator, kinetic_propagator, energy_fn, l = setup_sim()
    Lx = grid_boundaries[1]*2
    Ly = grid_boundaries[3]*2
    Lz = grid_boundaries[5]*2
    all_converged = True
    save_name = None
    if save_folder:
        save_name = save_folder+"/"+str(kx)
    guess1 = modulated_Gaussian(grid_shape, grid_boundaries, Lx/8, Ly/8, Lz/8, 2*np.pi*kx/Lx, 0, 0, 0)
    guess2 = modulated_Gaussian(grid_shape, grid_boundaries,  Lx/8, Ly/8, Lz/8, 2*np.pi*kx/Lx, 0, 0, 0)
    
    vm = 2
    start_time = time.time()
    grid, psi, max_iter, errs, mu1s, mu2s, Es, converged = run_simulation_ite(grid_shape,grid_boundaries,potential_propagator, kinetic_propagator,time_step,iterations,guess1, guess2, tolerance=tolerance,iter_check=500, save_name=save_name, energy_fn=energy_fn, noise_iters=noise_iters, animation=ani, vm=vm)
    # grid, psi = gpe.run_simulation_rte(grid_shape,grid_boundaries,potential_propagator, kinetic_propagator,time_step,iterations,guess1, guess2, plot_all=True)
    all_converged = all_converged and converged
    end_time = time.time()
    print("converged:", converged)
    print("elapsed time:", end_time-start_time)
    print("time per iteration:", (end_time-start_time)/(max_iter+1))
    psi1 = psi[0,:,:,:]
    psi2 = psi[1,:,:,:]
    n1 = np.abs(psi1)**2
    n2 = np.abs(psi2)**2
    # overlap = np.sum(n1*n2)/np.sqrt((np.sum(n1**2)*np.sum(n2**2)))
    # print("overlap:", overlap)
    plot_density_dual_species_int(grid,[n1.get(), n2.get()], data_labels=("n1", "n2"), save_name=str(kx)+"_xy.png" , save_folder=save_folder, sum_axes=("z",), show=False, units=(l*1e6, "um"))
    plot_density_dual_species_int(grid,[n1.get(), n2.get()], data_labels=("n1", "n2"), save_name=str(kx)+"_xz.png" , save_folder=save_folder, sum_axes=("y",), show=False, units=(l*1e6, "um"))
    plot_density_dual_species_int(grid,[n1.get(), n2.get()], data_labels=("n1", "n2"), save_name=str(kx)+"_x.png" , save_folder=save_folder, sum_axes=("y","z"), show=False, units=(l*1e6, "um"))
    plt.clf()
    plt.close("all")
    # indices = list(range(0,max_iter+1))
    # plt.plot(indices[1:], mu1s[1:], label="1")
    # plt.plot(indices[1:], mu2s[1:], label="2")
    # plt.legend()
    # plt.savefig(save_folder+"/"+r_str+"mus.png")
    # plt.close()
    # plt.plot(indices[1:], errs[1:])
    # plt.yscale("log")
    # plt.savefig(save_folder+"/"+r_str+"mus.png")
    # plt.close()
    # ys = grid[1]
    # ys = numpy.ravel(ys.get())
    E_components_final = Es[-1]
    E_final = E_components_final[-1]
    return E_final
        
kxs = 2**np.linspace(3, 8, 6)
Es = []
for kx in kxs:
    E = float(run_sim(kx))
    Es.append(E)

plt.plot(kxs.get(), Es)
plt.xlabel("kx (2pi/Lx)")
plt.ylabel("E")
plt.savefig(save_folder+"/Evk.png")
print(repr(kxs.get()))
print(repr(Es))
