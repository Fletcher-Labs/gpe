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

# read in simulation parameters
sim_name = sys.argv[1]
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
dip = args["dip"]
a12 = float(args["a12"])*a0
sag = float(args["sag"])*1e-6
Ny = int(args["Ny"])
tolerance = float(args["tol"])
iterations = int(args["iter"])
noise_iters = 0
ani = None
if "noise_iters" in args.keys():
    print("adding noise")
    noise_iters = int(args["noise_iters"])
if "ani" in args.keys():
    print("saving movie")
    ani = int(args["ani"])
save_folder = "results/"+sim_name
guess_file = save_folder+"/data.npz"
data = None
if os.path.exists(guess_file):
  data =  np.load(guess_file)
  psi = data["psi"]
  guess1 = psi[0,:,:,:]
  guess2 = psi[1,:,:,:]

# experimental parameters
wx = 2*np.pi*25
wy = 2*np.pi*5
wz = 2*np.pi*35
a1 = 83*a0
a2 = 100.4*a0
add = 65.5*a0
N1 = 1e5
N2 = 2e5
m1 = 166*1.66e-27
m2 = 87*1.66e-27
m12 = m1*m2/(m1+m2)
g11 = 4*np.pi*hbar**2*a1/m1
g22 = 4*np.pi*hbar**2*a2/m2
g12 = 2*np.pi*hbar**2*a12/m12
gdd = 4*np.pi*hbar**2*add/m1
pol_ratio = 1.8
dipole_direction = [0, 0, 1]

gratio = g11*g22/g12**2
prediction = "miscible" if gratio > 1 else "immiscible"
print(g11*g22/g12**2)
print(prediction)

# dimensionless units
l = np.sqrt(hbar/(m1*wx))
tau = 1/wx
print(l)
sag = sag/l

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
dipole_strength = gdd*N1/(hbar*wx*l**3)
d1 = gdd*N1/(hbar*wx*l**3)
d2 = gdd*N2/(hbar*wx*l**3)*(0.5)**2/(7)**2
d12 = gdd*N2/(hbar*wx*l**3)*(0.5)/(7)
d21 = gdd*N1/(hbar*wx*l**3)*(0.5)/(7)
dipole_strengths = [d1, d12, d21, d2]

# # simulation grid
Nx = 2**7
Nz = 2**7
grid_shape = [Nx, Ny, Nz]
time_step = 1e-2

Lx = 15
Ly = 55
Lz = 15
grid_boundaries = [-Lx/2, Lx/2, -Ly/2, Ly/2, -Lz/2, Lz/2]

save_name = None
if save_folder:
  save_name = save_folder+"/data"

if dip == "sd":
    potential_propagator = lambda psi, grid, kgrid, dt: single_dipolar_potential_propagator(psi, grid, kgrid, dt, gammas, anisotropy, pol_ratio, dipole_strength, dipole_direction, sag=sag)
    kinetic_propagator = lambda psi, grid, kgrid, dt: mixture_kinetic_propagator(psi, grid, kgrid, dt, mass_ratio)
    energy_fn = lambda psi, grid, kgrid: energy_functional_single_dipolar(psi,grid,kgrid,gammas,anisotropy, pol_ratio, mass_ratio, dipole_strength, dipole_direction, atom_nums, sag=sag)
elif dip == "dd":
    potential_propagator = lambda psi, grid, kgrid, dt: double_dipolar_potential_propagator(psi, grid, kgrid, dt, gammas, anisotropy, pol_ratio, dipole_strengths, dipole_direction, sag=sag)
    kinetic_propagator = lambda psi, grid, kgrid, dt: mixture_kinetic_propagator(psi, grid, kgrid, dt, mass_ratio)
    energy_fn = lambda psi, grid, kgrid: energy_functional_double_dipolar(psi,grid,kgrid,gammas,anisotropy, pol_ratio, mass_ratio, dipole_strengths, dipole_direction, atom_nums, sag=sag)
else:
    raise ValueError("dipolar potential type not valid")
    
if not data:
  guess1 = Gaussian_ansatz(grid_shape, grid_boundaries, 5, 10, 5, noise_level=0.4)
  guess2 = Gaussian_ansatz(grid_shape, grid_boundaries, 5, 10, 5, noise_level=0.4)
start_time = time.time()
grid, psi, max_iter, errs, mu1s, mu2s, Es, converged = run_simulation_ite(grid_shape,grid_boundaries,potential_propagator, kinetic_propagator,time_step,iterations,guess1, guess2, tolerance=tolerance,iter_check=500, plot_all=False, save_name=save_name, energy_fn=energy_fn, noise_iters=noise_iters, animation=ani)
# grid, psi = gpe.run_simulation_rte(grid_shape,grid_boundaries,potential_propagator, kinetic_propagator,time_step,iterations,guess1, guess2, plot_all=True)
end_time = time.time()
print("converged:", converged)
print("elapsed time:", end_time-start_time)
print("time per iteration:", (end_time-start_time)/(max_iter+1))
psi1 = psi[0,:,:,:]
psi2 = psi[1,:,:,:]
n1 = np.abs(psi1)**2
n2 = np.abs(psi2)**2
overlap = np.sum(n1*n2)/np.sqrt((np.sum(n1**2)*np.sum(n2**2)))
print("overlap:", overlap)
Cs, C = contrast(n1)
plot_density_unnormalized_int(grid,[n1.get(), n2.get()], data_labels=("n1", "n2"), save_name="xy.png" , save_folder=save_folder, sum_axes=("z",), show=True, units=(l*1e6, "um"))
plot_density_unnormalized_int(grid,[n1.get(), n2.get()], data_labels=("n1", "n2"), save_name="yz.png" , save_folder=save_folder, sum_axes=("x",), show=True, units=(l*1e6, "um"))

plt.clf()
plt.close("all")
iterations = list(range(0,max_iter+1))
plt.plot(iterations, mu1s, label="1")
plt.plot(iterations, mu2s, label="2")
plt.legend()
plt.savefig(save_folder+"/mus.png")
plt.close()
ys = grid[1]
ys = numpy.ravel(ys.get())
plt.plot(ys, Cs.get())
plt.savefig(save_folder+"/contrast.png")
print("contrast:", C)


