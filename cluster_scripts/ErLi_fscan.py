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
fx1 = float(args["fx1"])
fx2 = float(args["fx2"])
Nf = int(args["Nf"])
a12 = float(args["a12"])*a0

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


def setup_sim(fx):
    # experimental parameters
    wx = 2*np.pi*fx
    wy = 2*np.pi*150
    wz = 2*np.pi*150
    a1 = 65*a0
    a2 = 10*a0
    add = 64*a0
    N1 = 2e4
    N2 = 2e4
    m1 = 166*1.66e-27
    m2 = 7*1.66e-27
    m12 = m1*m2/(m1+m2)
    g11 = 4*np.pi*hbar**2*a1/m1
    g22 = 4*np.pi*hbar**2*a2/m2
    g12 = 2*np.pi*hbar**2*a12/m12
    gdd = 4*np.pi*hbar**2*add/m1
    pol_ratio = 270/166
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
    dipole_strengths = [d1, d12*1/14, d21*1/14, d2*(1/14)**2]
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

    gamma = sum(gammas)/len(gammas)
    mu = (15*gamma/(8*2**(3/2)*np.pi)*np.sqrt(anisotropy[0]*anisotropy[1]*anisotropy[2]))**(2/5)
    RTF = [np.sqrt(2*mu/anisotropy[0]), np.sqrt(2*mu/anisotropy[1]), np.sqrt(2*mu/anisotropy[2])]
    Lx = 3.5*RTF[0]
    Ly = 3.5*RTF[1]
    Lz = 3.5*RTF[2]
    print("Ls:", Lx, Ly, Lz)

    Nx = 2**9
    Ny = 2**6
    Nz = 2**6
    
    grid_shape = [Nx, Ny, Nz]
    grid_boundaries = [-Lx/2, Lx/2, -Ly/2, Ly/2, -Lz/2, Lz/2]

    potential_propagator = lambda psi, grid, kgrid, dt: double_dipolar_potential_cutoff_intra_LHY_propagator(psi, grid, kgrid, dt, gammas, anisotropy, pol_ratio, dipole_strengths, dipole_direction, LHY_factors, sag=0)
    kinetic_propagator = lambda psi, grid, kgrid, dt: mixture_kinetic_propagator(psi, grid, kgrid, dt, mass_ratio)
    energy_fn = lambda psi, grid, kgrid: energy_functional_double_dipolar_cutoff_intra_LHY(psi,grid,kgrid,gammas,anisotropy, pol_ratio, mass_ratio, dipole_strengths, dipole_direction, atom_nums, LHY_factors, sag=0)
    return grid_shape, grid_boundaries, potential_propagator, kinetic_propagator, energy_fn, l

def run_sim(runs, fx):
    grid_shape, grid_boundaries, potential_propagator, kinetic_propagator, energy_fn, l = setup_sim(fx)
    Lx = grid_boundaries[1]*2
    Ly = grid_boundaries[3]*2
    Lz = grid_boundaries[5]*2
    Es_final = []
    all_converged = True
    run_time_max = None
    Cout = []
    fs1out = []
    for r in range(runs):
        vm = 10
        save_name = None
        r_str = "run_"+str(r+start_run)+"_fx_" + str(int(fx)) 
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
        elif guess == "random_sym":
            guess1 = random_guess_sym(grid_shape, grid_boundaries)
            guess2 = random_guess_sym(grid_shape, grid_boundaries)
        elif guess == "perlin":
            res = (2**random.randint(0,6),2**random.randint(0,6),2**random.randint(0,6))
            guess1 = perlin_guess(grid_shape, grid_boundaries, res)
            guess2 = perlin_guess(grid_shape, grid_boundaries, res)
        elif guess == "prev":
            try:
                data =  np.load(save_name+".npz")
                psi = data["psi"]
                guess1 = psi[0,:,:,:]
                guess2 = psi[1,:,:,:]
            except ValueError:
                print("prev data not provided")
        else:
            raise ValueError("invalid guess")
        
        vm = 2
        print("starting run " + str(r+start_run))
        start_time = time.time()
        grid, psi, max_iter, errs, mu1s, mu2s, Es, converged = run_simulation_ite(grid_shape,grid_boundaries,potential_propagator, kinetic_propagator,time_step,iterations,guess1, guess2, tolerance=tolerance,iter_check=500, save_name=save_name, energy_fn=energy_fn, noise_iters=noise_iters, animation=ani, vm=vm)
        # grid, psi = gpe.run_simulation_rte(grid_shape,grid_boundaries,potential_propagator, kinetic_propagator,time_step,iterations,guess1, guess2, plot_all=True)
        all_converged = all_converged and converged
        end_time = time.time()
        run_time = end_time-start_time
        print("converged:", converged)
        print("elapsed time:", run_time)
        if r == 0 or run_time > run_time_max: run_time_max = run_time
        print("time per iteration:", (run_time)/(max_iter+1))
        psi1 = psi[0,:,:,:]
        psi2 = psi[1,:,:,:]
        n1 = np.abs(psi1)**2
        n2 = np.abs(psi2)**2
        # overlap = np.sum(n1*n2)/np.sqrt((np.sum(n1**2)*np.sum(n2**2)))
        # print("overlap:", overlap)
        plot_density_dual_species_int(grid,[n1.get(), n2.get()], data_labels=("n1", "n2"), save_name=r_str+"xy.png" , save_folder=save_folder, sum_axes=("z",), show=False, units=(l*1e6, "um"))
        plot_density_dual_species_int(grid,[n1.get(), n2.get()], data_labels=("n1", "n2"), save_name=r_str+"xz.png" , save_folder=save_folder, sum_axes=("y",), show=False, units=(l*1e6, "um"))
        plot_density_dual_species_int(grid,[n1.get(), n2.get()], data_labels=("n1", "n2"), save_name=r_str+"x.png" , save_folder=save_folder, sum_axes=("y","z"), show=False, units=(l*1e6, "um"))
        plt.clf()
        plt.close("all")
        Cs, C = contrast(n1)
        plt.plot(Cs.get())
        plt.title("Contrast = " + str(C))
        plt.savefig(save_name+"_Cs.png")
        try:
            fig, f1, f2 = leggett_superfluid_fraction(n1, n2)
            fig.savefig(save_name+"_domain_boundaries.png")
            plt.close(fig)
        except:
            f1 = 0
            f2 = 0
        print("superfluid fractions: ", f1, f2)
        
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
        Es_final.append(float(E_final))
        # write outputs to a csv file
        output_filename = save_folder+"/" + r_str + "_outputs.csv"
        outputs = [["E","C","fs1","fs2"],[float(E_final), C, f1, f2]]
        with open(output_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(outputs)
        Cout.append(float(C))
        fs1out.append(float(f1))
        
    if runs > 1:
        plt.plot(np.arange(start_run, runs+start_run).get(), Es_final)
        plt.title("Best run:"+ str(start_run+np.argmin(np.array(Es_final))) + ", Best energy: " + str(min(Es_final)))
        plt.savefig(save_folder+"/"+str(start_run)+"_final_Es.png")
        plt.close()
        if save_folder:
            for r in range(runs):
                r_str = "run_"+str(r+start_run)+"_fx_" + str(int(fx)) 
                save_name = save_folder+"/" + r_str 
                if r != np.argmin(np.array(Es_final)):
                    file_path = save_name+".npz"
                    if os.path.exists(file_path):
                        os.remove(file_path)
    return run_time_max, Cout, fs1out

fxs = np.linspace(fx1, fx2, Nf)
t0 = time.time()
not_finished = False
fxs_plot = []
Cs = []
fs1s = []
for i, fx in enumerate(fxs):
    run_time_max, Cout, fs1out = run_sim(runs, fx)
    t = time.time()
    fxs_plot += [float(fx)]*runs
    Cs += Cout
    fs1s += fs1out
    if runs*run_time_max*1.2+t > t0 + 120*60:
        not_finished = True
        break
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.scatter(fxs_plot, Cs, c="blue")
ax2.scatter(fxs_plot, fs1s, c="red")
ax1.set_xlabel("fx")
ax1.set_ylabel("Contrast", color="blue")
ax1.tick_params(axis='x', colors='blue')
ax2.tick_params(axis='x', colors='red')
ax2.set_ylabel("Superfluid Fraction", color="red")
fig.savefig(save_folder+"/superfluid.png")

print(not_finished)
if not_finished:
    new_run_filename = sim_name+".csv"
    new_run_name = sim_name
    fx1_new = fxs[i+1]
    fx2_new = fxs[-1]
    Nf_new = Nf-i-1
    new_sim_params = [["tol", "iter", "dt", "fx1", "fx2", "Nf","runs","guess","a12","ani","noise_iters"],[tolerance, iterations, time_step, fx1_new, fx2_new, Nf_new,runs,guess,a12/a0,ani,noise_iters]]
    with open(new_run_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_sim_params)
    os.system("cd gpe \n sbatch --job-name=" + new_run_name + " ErLi.sh")