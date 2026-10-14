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
file_name = sim_name+".csv"
with open(file_name, mode='r') as file:
    csvReader = csv.DictReader(file)
    for line in csvReader:
        args = line
print("read sim params:", args)
# physical constants
hbar = 1.05e-34 # J.s
a0 = 5.29e-11 # m
# inputs
tolerance = float(args["tol"])
iterations = int(args["iter"])
time_step = float(args["dt"])
runs = 1
noise_iters = 0
ani = None
perlin = False
start_run = 0
guess = "random"
if "noise_iters" in args.keys():
    print("adding noise")
    noise_iters = int(args["noise_iters"])
if "ani" in args.keys():
    print("saving movie")
    ani = int(args["ani"])
if "runs" in args.keys():
    print("doing multiple runs")
    runs = int(args["runs"])
if "start_run" in args.keys():
    start_run = int(args["start_run"])
if "guess" in args.keys():
    guess = args["guess"]


def setup_sim(N1):
    # experimental parameters
    wx = 2*np.pi*37.5*4e3/N1
    wy = 2*np.pi*110
    wz = 2*np.pi*150
    a1 =  100*a0
    a2 =  80*a0
    a12 =  100*a0
    add = 131*a0
    N2 = N1
    m = 164*1.66e-27
    m1=m2=m
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

    Lx = 30
    Ly = 2
    Lz = 2

    Nx = 2**9
    Ny = 2**6
    Nz = 2**6

    grid_shape = [Nx, Ny, Nz]
    grid_boundaries = [-Lx/2, Lx/2, -Ly/2, Ly/2, -Lz/2, Lz/2]

    potential_propagator = lambda psi, grid, kgrid, dt: double_dipolar_potential_cutoff_intra_LHY_propagator(psi, grid, kgrid, dt, gammas, anisotropy, pol_ratio, dipole_strengths, dipole_direction, LHY_factors, sag=0)
    kinetic_propagator = lambda psi, grid, kgrid, dt: mixture_kinetic_propagator(psi, grid, kgrid, dt, mass_ratio)
    energy_fn = lambda psi, grid, kgrid: energy_functional_double_dipolar_cutoff_intra_LHY(psi,grid,kgrid,gammas,anisotropy, pol_ratio, mass_ratio, dipole_strengths, dipole_direction, atom_nums, LHY_factors, sag=0)
    return grid_shape, grid_boundaries, potential_propagator, kinetic_propagator, energy_fn, l

def run_sim(runs, N1):
    grid_shape, grid_boundaries, potential_propagator, kinetic_propagator, energy_fn, l = setup_sim(N1)
    Lx = grid_boundaries[1]*2
    Ly = grid_boundaries[3]*2
    Lz = grid_boundaries[5]*2
    Es_final = []
    all_converged = True
    for r in range(runs):
        vm = 1
        save_name = None
        r_str = "run_"+str(r+start_run)+"_N1_" + str(N1/1e3) if runs > 1 else "N1_" + str(N1/1e3)
        if save_folder:
            save_name = save_folder+"/" + r_str 
        if guess == "gauss":
            guess1 = Gaussian_ansatz(grid_shape, grid_boundaries, random.random()*Lx/2, random.random()*Ly/2, random.random()*Lz/2, 0.2)
            guess2 = Gaussian_ansatz(grid_shape, grid_boundaries,  random.random()*Lx/2, random.random()*Ly/2, random.random()*Lz/2, 0.2)
        elif guess == "gauss_sym":
            s1 = random.random()*Lx/2
            s2 =  random.random()*Lx/2
            guess1 = Gaussian_ansatz(grid_shape, grid_boundaries, s1, s1, s1, 0.2)
            guess2 = Gaussian_ansatz(grid_shape, grid_boundaries, s2, s2, s2, 0.2)
        elif guess == "random":
            guess1 = random_guess(grid_shape, grid_boundaries)
            guess2 = random_guess(grid_shape, grid_boundaries)
        elif guess == "perlin":
            res = (2**random.randint(0,6),2**random.randint(0,6),2**random.randint(0,6))
            guess1 = perlin_guess(grid_shape, grid_boundaries, res)
            guess2 = perlin_guess(grid_shape, grid_boundaries, res)
        else:
            raise ValueError("invalid guess")
        vm = 5
        print("starting run " + str(r+start_run))
        start_time = time.time()
        grid, psi, max_iter, errs, mu1s, mu2s, Es, converged = run_simulation_ite(grid_shape,grid_boundaries,potential_propagator, kinetic_propagator,time_step,iterations,guess1, guess2, tolerance=tolerance,iter_check=100, save_name=save_name, energy_fn=energy_fn, noise_iters=noise_iters, animation=ani, vm=vm)
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
        plot_density_unnormalized_int(grid,[n1.get(), n2.get()], data_labels=("n1", "n2"), save_name=r_str+"xy.png" , save_folder=save_folder, sum_axes=("z",), show=False, units=(l*1e6, "um"))
        plot_density_unnormalized_int(grid,[n1.get(), n2.get()], data_labels=("n1", "n2"), save_name=r_str+"xz.png" , save_folder=save_folder, sum_axes=("y",), show=False, units=(l*1e6, "um"))
        plot_density_unnormalized_int(grid,[n1.get(), n2.get()], data_labels=("n1", "n2"), save_name=r_str+"x.png" , save_folder=save_folder, sum_axes=("y","z"), show=False, units=(l*1e6, "um"))
        plt.clf()
        plt.close("all")
        indices = list(range(0,max_iter+1))
        plt.plot(indices[1:], mu1s[1:], label="1")
        plt.plot(indices[1:], mu2s[1:], label="2")
        plt.legend()
        plt.savefig(save_folder+"/"+r_str+"mus.png")
        plt.close()
        plt.plot(indices[1:], errs[1:])
        plt.yscale("log")
        plt.savefig(save_folder+"/"+r_str+"mus.png")
        plt.close()
        # ys = grid[1]
        # ys = numpy.ravel(ys.get())
        E_components_final = Es[-1]
        E_final = E_components_final[-1]
        Es_final.append(float(E_final))
        
    if runs > 1:
        plt.plot(np.arange(start_run, runs+start_run).get(), Es_final)
        plt.title("Best run:"+ str(start_run+np.argmin(np.array(Es_final))) + ", Best energy: " + str(min(Es_final)))
        plt.savefig(save_folder+"/"+str(start_run)+"_final_Es.png")
        plt.close()
        if save_folder:
            for r in range(runs):
                r_str = "run_"+str(r+start_run)+"_N1_" + str(N1/1e3) if runs > 1 else "N1_" + str(N1/1e3)
                save_name = save_folder+"/" + r_str 
                if r != np.argmin(np.array(Es_final)):
                    file_path = save_name+".npz"
                    if os.path.exists(file_path):
                        os.remove(file_path)
        
Ns = np.linspace(4,10,6)*1e3
for N in Ns:
    run_sim(runs, N)