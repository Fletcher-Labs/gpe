from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.optimize import curve_fit
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
a12i = float(args["a12i"])
a12f = float(args["a12f"])
Na = int(args["Na"])
N2i = int(args["N2i"])
N2f = int(args["N2f"])
N2N = int(args["N2N"])
N1 = int(args["N1"])
fyz = float(args["fz"])
fx = float(args["fx"])
lx = float(args["lx"])

runs = 1
noise_iters = 0
ani = None
perlin = False
start_run = 0
save_data = False
guess = "random_sym"
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
if "save_data" in args.keys():
    save_data = bool(args["save_data"])

if ani == 0: ani = None
pad_fraction = 0.35
Nx = 2**7
Ny = 2**6
Nz = 2**6
Ly = 20
Lz = 20
Lx = lx/(1-pad_fraction)
print("Ls:", Lx, Ly, Lz)
dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
dLs = [dLx, dLy, dLz]

def setup_sim(a12, N2):
    # experimental parameters
    wyz = 2*np.pi*fyz
    a1 = 68*a0
    add = 65*a0
    a2 = 80*a0
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
    l = np.sqrt(hbar/(m1*wyz))
    tau = 1/wyz
    print(l)

    # scaled parameters
    atom_nums = [N1, N2]
    mass_ratio = m2/m1
    gamma11 = N1*g11/(hbar*wyz*l**3)
    gamma12 =  N2*g12/(hbar*wyz*l**3)
    gamma21 = N1*g12/(hbar*wyz*l**3)
    gamma22 =  N2*g22/(hbar*wyz*l**3)
    gammas =  [gamma11, gamma12, gamma21, gamma22]
    print(gammas)
    print(gamma11*gamma22/(gamma12*gamma21))
    print("gdd:",gdd)
    d1 = gdd*N1/(hbar*wyz*l**3)
    d2 = gdd*N2/(hbar*wyz*l**3)
    d12 = gdd*N2/(hbar*wyz*l**3)
    d21 = gdd*N1/(hbar*wyz*l**3)
    dipole_strengths = [d1, d12*1/14, d21*1/14, d2*(1/14)**2]
    edd1 = d1/gamma11
    edd2 = d2/gamma22
    f_ratio = fx/fyz

    # simulation grid
    grid_shape = [Nx, Ny, Nz]
    grid_boundaries = [-Lx/2, Lx/2, -Ly/2, Ly/2, -Lz/2, Lz/2]
    potential_propagator = lambda psi, grid, kgrid, dt: dipole_potential_1D_box_harmonic_propagator(psi, grid, kgrid, dt, gammas, pol_ratio, f_ratio, dipole_strengths, dipole_direction, pad_fraction)
    kinetic_propagator = lambda psi, grid, kgrid, dt: box_1D_kinetic_propagator(psi, grid, kgrid, dt, mass_ratio, pad_fraction)
    energy_fn = lambda psi, grid, kgrid: energy_functional_double_dipolar_cutoff(psi,grid,kgrid,gammas,[0,1,1], pol_ratio, mass_ratio, dipole_strengths, dipole_direction, atom_nums)
    return grid_shape, grid_boundaries, potential_propagator, kinetic_propagator, energy_fn, l

def run_sim(runs, a12, N2):
    grid_shape, grid_boundaries, potential_propagator, kinetic_propagator, energy_fn, l = setup_sim(a12, N2)
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
        r_str = "N2_"+str(N2/1e3)+"_a12_" + str(int(a12/a0)) + "_run_"+str(r+start_run)
        if save_folder:
            save_name = save_folder+"/" + r_str 
        # calculate appropriate initial guesses
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
        # run a simulation
        vm = 2
        print("starting run " + str(r+start_run))
        start_time = time.time()
        grid, psi, max_iter, errs, mu1s, mu2s, Es, converged = run_simulation_ite(grid_shape,grid_boundaries,potential_propagator, kinetic_propagator,time_step,iterations,guess1, guess2, tolerance=tolerance,iter_check=500, save_name=save_name, energy_fn=energy_fn, noise_iters=noise_iters, animation=ani, vm=vm, save_data=save_data)
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
        # density extraction
        n1 = N1*np.abs(psi1)**2
        n2 = N2*np.abs(psi2)**2
        overlap = np.sum(n1*n2)/np.sqrt((np.sum(n1**2)*np.sum(n2**2)))
        print("overlap:", overlap)
        # gaussian fitting of density cuts
        n1z = n1[int(Ny/2),int(Nx/2),:].get()
        n2z = n2[int(Ny/2),int(Nx/2),:].get()
        n1y = n1[:,int(Nx/2),int(Nz/2)].get()
        n2y = n2[:,int(Nx/2),int(Nz/2)].get()
        lz1 = lz2 = ly1 = ly2 = 0
        # gauss = lambda x, x0, amp, sigma: amp*numpy.exp(-(x-x0)**2/sigma**2) 
        # z = numpy.linspace(-Lz/2, Lz/2, num=Nz)
        # popt1, _ = curve_fit(gauss, z, n1z)
        # popt2, _ = curve_fit(gauss, z, n2z)
        # plt.clf()
        # plt.plot(z, n1z)
        # plt.plot(z, n2z)
        # plt.plot(z, gauss(z,*popt1))
        # plt.plot(z, gauss(z,*popt2))
        # lz1 = popt1[-1]
        # lz2 = popt2[-1]
        # print("lz1: ", popt1[-1])
        # print("lz2: ", popt2[-1])
        # # plt.savefig(save_name+"gaussz.png")
        # plt.show()
        # y = numpy.linspace(-Ly/2, Ly/2, num=Nz)
        # popt1, _ = curve_fit(gauss, y, n1y)
        # popt2, _ = curve_fit(gauss, y, n2y)
        # plt.clf()
        # plt.plot(y, n1y)
        # plt.plot(y, n2y)
        # plt.plot(y, gauss(y,*popt1))
        # plt.plot(y, gauss(y,*popt2))
        # ly1 = popt1[-1]
        # ly2 = popt2[-1]
        # print("ly1: ", popt1[-1])
        # print("ly2: ", popt2[-1])
        # plt.savefig(save_name+"gaussy.png")
        plt.show()
        plot_density_dual_species_int(grid,[n1.get(), n2.get()], data_labels=("n1", "n2"), save_name=r_str+"xy.png" , save_folder=save_folder, sum_axes=("z",), show=False, units=(l*1e6, "um"), dLs=dLs)
        plot_density_dual_species_int(grid,[n1.get(), n2.get()], data_labels=("n1", "n2"), save_name=r_str+"xz.png" , save_folder=save_folder, sum_axes=("y",), show=False, units=(l*1e6, "um"), dLs=dLs)
        plot_density_dual_species_int(grid,[n1.get(), n2.get()], data_labels=("n1", "n2"), save_name=r_str+"x.png" , save_folder=save_folder, sum_axes=("y","z"), show=False, units=(l*1e6, "um"), dLs=dLs)
        plot_density_dual_species_int(grid,[n1.get(), n2.get()], data_labels=("n1", "n2"), save_name=r_str+"z.png" , save_folder=save_folder, sum_axes=("x","y"), show=False, units=(l*1e6, "um"), dLs=dLs)
        plt.clf()
        plt.close("all")
        n_total = (n1+n2)/l**3
        peak_n = np.max(n_total)
        try:
            Cs, C = contrast(n1)
            plt.plot(Cs.get())
            plt.title("Contrast = " + str(C))
            plt.savefig(save_name+"_Cs.png")
        except:
            C = 0
        f1, f2 = leggett_superfluid_fraction(n1, n2)
        print("superfluid fractions: ", f1, f2)
        E_components_final = Es[-1]
        E_std = [Es[-i][-1] for i in range(1,10)]
        dE = np.std(np.array(E_std))
        E_final = E_components_final[-1]
        Es_final.append(float(E_final))
        # write outputs to a csv file
        output_filename = save_folder+"/" + r_str + "_outputs.csv"
        outputs = [["E","dE", "C","peak_n", "fs1","fs2","lz1","lz2","ly1","ly2"],[float(E_final),float(dE), C, float(peak_n), f1, f2, lz1, lz2, ly1, ly2]]
        with open(output_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(outputs)
        Cout.append(float(C))
        fs1out.append(float(f1))
    
    # if saving data to npz file, only keep data from lowest energy run
    if runs > 1:
        if save_folder:
            for r in range(runs):
                r_str = "N2_"+str(N2/1e3)+"_a12_" + str(int(a12/a0)) + "_run_"+str(r+start_run)
                save_name = save_folder+"/" + r_str 
                if r != np.argmin(np.array(Es_final)):
                    file_path = save_name+".npz"
                    if os.path.exists(file_path):
                        os.remove(file_path)
    return run_time_max, Cout, fs1out

a12s = np.linspace(a12i, a12f, Na)*a0
N2s = np.linspace(N2i, N2f, N2N,dtype=int)
t0 = time.time()
not_finished = False
enough_time = True
# perform scan
for j, N2 in enumerate(N2s):
    a12s_plot = []
    Cs = []
    fs1s = []
    if not enough_time:
        break
    for i, a12 in enumerate(a12s):
        run_time_max, Cout, fs1out = run_sim(runs, a12, N2)
        t = time.time()
        a12s_plot += [float(a12/a0)]*runs
        Cs += Cout
        fs1s += fs1out
        # check if there is enough time to keep running
        if runs*run_time_max*1.2+t > t0 + 120*60:
            not_finished = True
            enough_time = False
            break
    # plot results
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter(a12s_plot, Cs, c="blue")
    ax2.scatter(a12s_plot, fs1s, c="red")
    ax1.set_xlabel("a12")
    ax1.set_ylabel("Contrast", color="blue")
    ax1.tick_params(axis='x', colors='blue')
    ax2.tick_params(axis='x', colors='red')
    ax2.set_ylabel("Superfluid Fraction", color="red")
    fig.savefig(save_folder+"/N2_"+str(N2/1e3)+"_superfluid.png")
    plt.show()

print(not_finished)
## send new jobs if scan was not completed in time
if not_finished:
    new_run_filename = sim_name+".csv"
    new_run_name = sim_name
    if j < len(N2s):
        N2i_new = N2s[j]
        N2f_new = N2s[-1]
        N2N_new = N2N-j
        if ani is None: ani = 0
        new_sim_params = [["tol", "iter", "dt", "a12i", "a12f", "Na","lx", "fx", "fz","runs","guess","ani","noise_iters","N1","N2i","N2f","N2N"],[tolerance, iterations, time_step, a12i, a12f, Na,lx,fx,fyz,runs,guess,ani,noise_iters,N1,N2i_new, N2f_new, N2N_new]]
        with open(new_run_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(new_sim_params)
        os.system("cd gpe \n sbatch --job-name=" + new_run_name + " ErLi.sh aNscan_1D_box")
    if i < len(a12s)-1:
        N2_cont = N2s[j-1]
        a12i_new = a12s[i+1]/a0
        a12f_new = a12s[-1]/a0
        Na_new = Na-i-1
        if ani is None: ani = 0
        new_sim_params = [["tol", "iter", "dt", "a12i", "a12f", "Na","lx", "fx","fz", "runs","guess","ani","noise_iters","N1","N2i","N2f","N2N"],[tolerance, iterations, time_step, a12i_new, a12f_new, Na_new,lx,fx,fyz,runs,guess,ani,noise_iters,N1,N2_cont,N2_cont,1]]
        new_run_filename = sim_name+"_fill.csv"
        new_run_name = sim_name+"_fill"
        with open(new_run_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(new_sim_params)
        os.system("cd gpe \n sbatch --job-name=" + new_run_name + " ErLi.sh aNscan_1D_box")
else:   
    os.rename(file_name, save_folder+"/"+file_name)