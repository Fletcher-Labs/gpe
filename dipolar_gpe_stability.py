gpu = False
import numpy as np
from matplotlib import pyplot as plt
import gpe_functions_3D as gpe
import analysis_fns as plt_fns
import time
if gpu:
    import cupy as np
import pyfftw.interfaces as fftw

# physical constants
hbar = 1.05e-34 # J.s
a0 = 5.29e-11 # m

# simulation grid
Nx = 2**7
Ny = 2**7
Nz = 2**7
grid_shape = [Nx, Ny, Nz]

# experimental parameters
add = 134*a0 
m = 164*1.66e-27
gdd = 4*np.pi*hbar**2*add/m
w_avg = 2*np.pi*100

# quantities being scanned
trap_ratios_array = [np.logspace(np.log10(0.4), np.log10(7), 10), np.logspace(np.log10(0.4), np.log10(7), 10)]
trap_ratios_array = [[5]]
ascats = np.linspace(0.1*a0, add, 100)
atom_nums = np.array([1e4])
acrit_array = []
# simulation parameters 
Nx = 2**6
Ny = 2**6
Nz = 2**6
grid_shape = [Nx, Ny, Nz]
tolerance = 1e-6
Lx = 5
Ly = 5
Lz = 5
dt = 1e-1

# scan over atom number and trap ratio and find acrit
for i, atom_num in enumerate(atom_nums):
    acrit_vs_trap_ratio = []
    trap_ratios = trap_ratios_array[i]
    last_acrit_index = len(ascats)  # assume crit ascat decreases with trap ratio
    for trap_ratio in trap_ratios:
        wx =  w_avg/trap_ratio**(1/3)
        l = np.sqrt(hbar/(m*wx))
        # do a binary search to find the critical ascat 
        stable_vals = []
        done = False
        checking_list = ascats[:last_acrit_index] 
        while len(checking_list) > 0:
            # check middle of list 
            index = len(checking_list)//2 
            ascat = checking_list[index]
            g =  4*np.pi*hbar**2*ascat/m
            epsdd = add/ascat
            gamma = atom_num*g/(hbar*wx*l**3)
            dipole_strength = gdd*atom_num/(hbar*wx*l**3)
            anisotropy = [1,1,trap_ratio**2]
            # simulation parameters
            mu_TF =(15*gamma/(8*2**(3/2)*np.pi)*np.sqrt(anisotropy[0]*anisotropy[1]*anisotropy[2]))**(2/5)
            mu_TF_dip =(15*dipole_strength/(8*2**(3/2)*np.pi)*np.sqrt(anisotropy[0]*anisotropy[1]*anisotropy[2]))**(2/5)
            time_step = min([dt/mu_TF,dt/mu_TF_dip])
            R_TF = np.sqrt(2*mu_TF)
            R_z_TF = R_TF/trap_ratio
            grid_boundaries = [-Lx/2*R_TF, Lx/2*R_TF, -Ly/2*R_TF, Ly/2*R_TF, -Lz/2*R_z_TF, Lz/2*R_z_TF]
            print(ascat/a0, trap_ratio, atom_num)
            iterations = 100
            tolerance = 1e-10
            potential_fn = lambda psi, grid, kgrid: gpe.cylindrical_dipolar_potential(psi,grid,kgrid,gamma,trap_ratio,dipole_strength)
            guess = gpe.Gaussian_ansatz_TF(grid_shape, grid_boundaries,np.abs(gamma),anisotropy)
            iter_check=50
            grid, psi, max_iter, err, mu, conv = gpe.run_simulation_ite(grid_shape,grid_boundaries,potential_fn,time_step,iterations,guess,tolerance=tolerance,iter_check=iter_check,print_all=True)
            plt_fns.plot_density_unnormalized_int(grid,np.abs(psi)**2,sum_axes=("y"))
            iters = np.arange(0,max_iter+1,1)
            plt.scatter(iters, mu)
            plt.show()
            plt.scatter(iters, err)
            plt.yscale("log")
            plt.show()
            # if stable, check lower for unstable
            if conv:
              checking_list = checking_list[:len(checking_list)//2]
              stable_vals.append(float(ascat))
            # if unstable, check upper for stable
            else:
              checking_list = checking_list[len(checking_list)//2+1:]
        acrit_vs_trap_ratio.append(min(stable_vals))
        last_acrit_index = int(min(len(ascats), np.where(ascats==min(stable_vals))[0]+2))
    acrit_array.append(acrit_vs_trap_ratio)

# plot result
for i, atom_num in enumerate(atom_nums):
    acrits = acrit_array[i]
    print(atom_num)
    print(trap_ratios)
    print(acrits)
    plt.scatter(trap_ratios, np.array(acrits)/a0, label=f"N = {atom_num:.2E}" )
plt.xscale("log")
plt.legend()
plt.show()
