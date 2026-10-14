gpu = False
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import gpe_functions_3D as gpe
import analysis_fns as plt_fns
import time
from scipy import fft as fftw
from tqdm import tqdm

if gpu:
    import cupy as np
# import pyfftw.interfaces as fftw

# physical constants
hbar = 1.05e-34 # J.s
a0 = 5.29e-11 # m

# experimental parameters
wx = 2*np.pi*57
wy = 2*np.pi*57
wz = 2*np.pi*334
trap_ratio = wz/wx
add = 66*a0
ascat = 70*a0
Natoms = 4e4
m = 166*1.66e-27
g = 4*np.pi*hbar**2*ascat/m
epsdd = add/ascat
gdd = 4*np.pi*hbar**2*add/m
Q5edd = 1+3*epsdd**2/2
# y = (1-epsdd)/(3*epsdd)+0j
# Q5edd = (3*epsdd)**(5/2)/48*((8+26*y+33*y**2)*np.sqrt(1+y) + 15*y**3*np.log((1+np.sqrt(1+y))/np.sqrt(y)))
# assert (np.real(Q5edd) > 10*np.imag(Q5edd))
# Q5edd = np.real(Q5edd)
# print(Q5edd)

gLHY = (32/(3*np.sqrt(np.pi)))*g*ascat**(3/2)*Q5edd

# dimensionless units
l = np.sqrt(hbar/(m*wx))
tau = 1/wx


# from variational 
# scaled parameters
gamma = Natoms*g/(hbar*wx*l**3)
dipole_strength = gdd*Natoms/(hbar*wx*l**3)
LHY_strength = gLHY*Natoms**(3/2)/(hbar*wx*l**(9/2))
anisotropy = [1, (wy/wx)**2, (wz/wx)**2]
dipole_direction = [0, 0, 1]

# # simulation grid
Nx = 2**7
Ny = 2**7
Nz = 2**7
grid_shape = [Nx, Ny, Nz]

iterations = 1000
tolerance = 1e-8

mu_TF =(15*gamma/(8*2**(3/2)*np.pi)*np.sqrt(anisotropy[0]*anisotropy[1]*anisotropy[2]))**(2/5)
mu_TF_dip =(15*dipole_strength/(8*2**(3/2)*np.pi)*np.sqrt(anisotropy[0]*anisotropy[1]*anisotropy[2]))**(2/5)
R_TF = np.sqrt(2*mu_TF)
R_z_TF = R_TF/trap_ratio
time_step = 1e-2

Lx = 15
Ly = 15
Lz = 15
grid_boundaries = [-Lx/2, Lx/2, -Ly/2, Ly/2, -Lz/2, Lz/2]

potential_fn = lambda psi, grid, kgrid: gpe.dipolar_potential(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction)
# potential_fn = lambda psi, grid, kgrid: gpe.harmonic_potential(psi,grid,kgrid,gamma,anisotropy)
energy_fn = lambda psi, grid, kgrid: gpe.energy_functional_dipole_LHY(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction,LHY_strength)
energy_fn = None

guess = gpe.Gaussian_ansatz(grid_shape, grid_boundaries, 0.5, 0.55, 2.55, 0.1)
start_time = time.time()
save_folder="results/"
save_name = save_folder+"Er_condensate"
grid, psi, max_iter_LHY, errs, mus, converged, Es = gpe.run_simulation_ite(grid_shape,grid_boundaries,potential_fn,time_step,iterations,guess,tolerance=tolerance,iter_check=100, energy_fn=energy_fn, save_name=save_name)
end_time = time.time()
print("elapsed time:", end_time-start_time, "s")
print("converged:", converged)
# print("energy:", np.real(Es[-1][-1]))
psi_fft = fftw.fftshift(fftw.fftn(psi))
xmin, xmax, ymin, ymax, zmin, zmax = grid_boundaries
kx = 2*np.pi*fftw.fftshift(fftw.fftfreq(Nx)*Nx/(xmax-xmin))
ky =  2*np.pi*fftw.fftshift(fftw.fftfreq(Ny)*Ny/(ymax-ymin))
kz =  2*np.pi*fftw.fftshift(fftw.fftfreq(Nz)*Nz/(zmax-zmin))
kgrid = np.meshgrid(kx,ky,kz,sparse=True) 

# plot result
n_sim = np.abs(psi)**2
n_TF = gpe.thomas_fermi_density (grid_shape, grid_boundaries, gamma, anisotropy)

# n_dipolar_TF = gpe.thomas_fermi_cyl_dipole (grid_shape, grid_boundaries, epsdd, gamma,trap_ratio)

plt_fns.plot_density_unnormalized_int(grid,n_sim, sum_axes=("z",), save_name="0_xy_density.png", save_folder=save_folder, data_labels=("n",), units=(l*1e6, "um"))
plt_fns.plot_density_unnormalized_int(grid,n_sim,sum_axes=("y",), save_name="0_xz_density.png", save_folder=save_folder, data_labels=("n",), units=(l*1e6, "um"))
plt_fns.plot_density_unnormalized_int(kgrid, np.abs(psi_fft)**2, sum_axes=("z",), data_labels=("n_FFT",))
plt_fns.plot_density_unnormalized_int(kgrid, np.abs(psi_fft)**2,  sum_axes=("y",), data_labels=("n_FFT",))

plt_fns.plot_density_unnormalized_int(grid,[n_sim,n_TF],sum_axes=("y","z"), data_labels=("GPE","TF"), save_name="x_density.png", save_folder=save_folder, units=(l*1e6, "um"))
plt_fns.plot_density_unnormalized_int(grid,[n_sim,n_TF],sum_axes=("x","y"), data_labels=("GPE","TF"), save_name="z_density.png", save_folder=save_folder, units=(l*1e6, "um"))

# plot convergence
iters_LHY = np.arange(0,max_iter_LHY+1,1)
plt.plot(iters_LHY, errs, label="LHY")
plt.title("Error")
plt.yscale('log')
plt.legend()
plt.show()

plt.plot(iters_LHY, mus, label="LHY")
plt.title("Chemical Potential")
plt.legend()
plt.show()