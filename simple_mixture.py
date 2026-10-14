gpu = False
import numpy as np
from matplotlib import pyplot as plt
import gpe_mixture as gpe
import analysis_fns as plt_fns
import time
import os
import gpe_functions_3D as gpe_single

if gpu:
    import cupy as np
# import pyfftw.interfaces as fftw

# physical constants
hbar = 1.05e-34 # J.s
a0 = 5.29e-11 # m

def run_sim(a12, save_folder):
    # experimental parameters
    wx = 2*np.pi*20
    wy = 2*np.pi*20
    wz = 2*np.pi*30
    a1 = 120*a0
    a2 = 100.4*a0
    add = 65.5*a0
    N1 = 1e5
    N2 = 0.01e5
    m1 = 166*1.66e-27
    m2 = 87*1.66e-27
    m12 = m1*m2/(m1+m2)
    g11 = 4*np.pi*hbar**2*a1/m1
    g22 = 4*np.pi*hbar**2*a2/m2
    g12 = 2*np.pi*hbar**2*a12/m12
    gdd = 4*np.pi*hbar**2*add/m1
    pol_ratio = (1.8)**2*m2/m1
    dipole_direction = [0, 0, 1]
    sag = 1.7e-6

    gratio = g11*g22/g12**2
    prediction = "miscible" if gratio > 1 else "immiscible"
    # print(g11*g22/g12**2)
    print(prediction)

    # dimensionless units
    l = np.sqrt(hbar/(m1*wx))
    tau = 1/wx
    # print(l)
    sag = sag/l

    # scaled parameters
    anisotropy = [1, (wy/wx)**2, (wz/wx)**2]
    mass_ratio = m2/m1
    gamma11 = N1*g11/(hbar*wx*l**3)
    gamma12 =  N2*g12/(hbar*wx*l**3)
    gamma21 = N1*g12/(hbar*wx*l**3)
    gamma22 =  N2*g22/(hbar*wx*l**3)
    gammas =  [gamma11, gamma12, gamma21, gamma22]
    print("gammas:", gammas)
    # print(gamma11*gamma22/(gamma12*gamma21))
    # dipole_strength = gdd*N1/(hbar*wx*l**3)
    d1 = gdd*N1/(hbar*wx*l**3)
    d2 = gdd*N2/(hbar*wx*l**3)*(0.5)**2/(7)**2
    d12 = gdd*N2/(hbar*wx*l**3)*(0.5)/(7)
    d21 = gdd*N1/(hbar*wx*l**3)*(0.5)/(7)
    dipole_strengths = [d1, d12, d21, d2]
    atom_nums = [N1, N2]

    # # simulation grid
    Nx = 2**7
    Ny = 2**7
    Nz = 2**7
    grid_shape = [Nx, Ny, Nz]

    iterations = 1000
    tolerance = 1e-6

    time_step = 1e-2

    Lx = 20
    Ly = 55
    Lz = 20
    grid_boundaries = [-Lx/2, Lx/2, -Ly/2, Ly/2, -Lz/2, Lz/2]
    dLx, dLy, dLz = Lx/Nx, Ly/Ny, Lz/Nz

    print("Ls:", l*Lx, l*Ly, l*Lz)
    # print(l*Lx/Nx, l*Ly/Ny, l*Lz/Nz)
    print("dt:", tau*time_step)

    # save_name = save_folder+"/data"
    save_name = None

    potential_propagator = lambda psi, grid, kgrid, dt: gpe.double_dipolar_potential_propagator(psi, grid, kgrid, dt, gammas, anisotropy, pol_ratio, dipole_strengths, dipole_direction, sag=sag)
    kinetic_propagator = lambda psi, grid, kgrid, dt: gpe.mixture_kinetic_propagator(psi, grid, kgrid, dt, mass_ratio)
    energy_fn = lambda psi, grid, kgrid: gpe.energy_functional_double_dipolar(psi,grid,kgrid,gammas,anisotropy, pol_ratio, mass_ratio, dipole_strengths, dipole_direction, atom_nums, sag=sag)

    guess1 = gpe.Gaussian_ansatz(grid_shape, grid_boundaries, 20, 50, 20, noise_level=0)
    guess2 = gpe.Gaussian_ansatz(grid_shape, grid_boundaries, 20, 50, 20, noise_level=0)
    start_time = time.time()
    grid, psi, max_iter, errs, mu1s, mu2s, Es, converged = gpe.run_simulation_ite(grid_shape,grid_boundaries,potential_propagator, kinetic_propagator,time_step,iterations,guess1, guess2, tolerance=tolerance,iter_check=50, plot_all=False, save_name=save_name, energy_fn=energy_fn, animation=None)
    # grid, psi = gpe.run_simulation_rte(grid_shape,grid_boundaries,potential_propagator, kinetic_propagator,time_step,iterations,guess1, guess2, plot_all=True)
    end_time = time.time()
    print("elapsed time:", end_time-start_time, "s")
    print("converged:", converged)
    psi1 = psi[0,:,:,:]
    psi2 = psi[1,:,:,:]
    n1 = np.abs(psi1)**2
    n2 = np.abs(psi2)**2
    overlap = np.sum(n1*n2)/np.sqrt((np.sum(n1**2)*np.sum(n2**2)))
    print("overlap:", overlap)
    Cs, C = gpe.contrast(n1)
    print("contrast:", C)
    plt_fns.plot_density_unnormalized_int(grid,[np.abs(psi[0,:,:,:])**2,np.abs(psi[1,:,:,:])**2], data_labels=("n1", "n2"), save_name="xy.png" , save_folder=save_folder, sum_axes=("z",), show=True, units=(l*1e6, "um"))
    plt_fns.plot_density_unnormalized_int(grid,[np.abs(psi[0,:,:,:])**2,np.abs(psi[1,:,:,:])**2], data_labels=("n1", "n2"), save_name="yz.png" , save_folder=save_folder, sum_axes=("x",), show=True, units=(l*1e6, "um"))
    n_TF = gpe_single.thomas_fermi_cyl_dipole(grid_shape, grid_boundaries, add/a1, gamma11, wz/wx)
    plt_fns.plot_density_unnormalized_int(grid,[n1, n_TF], data_labels=("n1", "nTF"), save_name="xy.png" , save_folder=save_folder, sum_axes=("z","x"), show=True, units=(l*1e6, "um"))
    plt.close()
    # iterations = np.arange(0,max_iter+1,1)
    # plt.plot(iterations[1:], mu1s[1:], label="1")
    # plt.plot(iterations[1:], mu2s[1:], label="2")
    # plt.legend()
    # plt.show()
    return Cs, grid

ws = [100, 50, 25, 10, 5]

def miscibility_scan(ws, save_folder):
    for w in ws:
        a12s = a0/np.linspace(1/30,1/5, 20)
        overlaps = []
        gratios = []
        save_folder = save_folder + str(w) + "/" 
        for a12 in a12s:
            gratio, overlap = run_sim(a12, w, save_folder)
            gratios.append(gratio)
            overlaps.append(overlap)
        
        plt.scatter(gratios, overlaps)
        plt.xlabel("g ratio")
        plt.ylabel("Overlap integral")
        plt.savefig(os.path.join(save_folder, "overlaps.png"))
        np.savetxt(os.path.join(save_folder, "gratios.csv"), gratios, delimiter=',')
        np.savetxt(os.path.join(save_folder, "overlaps.csv"), overlaps, delimiter=',')
        plt.close()

    plt.axvline(1,color="gray", linestyle="--")
    for w in ws:
        gratios = np.loadtxt(save_folder+"gratios.csv", delimiter=",")
        overlaps = np.loadtxt(save_folder+"overlaps.csv", delimiter=",")
        plt.scatter(gratios, overlaps, label="w="+str(w))
    plt.xlabel("g11g22/g12^2")
    plt.ylabel("Overlap Integral")
    plt.legend()
    plt.show()

c, grid = run_sim(0.1*a0, save_folder=None)
# y = grid[1]
# plt.plot(np.ravel(y), c)
# plt.show()
# c = run_sim(70*a0, save_folder="results/70_adss_dt1e-2")
# print(c)