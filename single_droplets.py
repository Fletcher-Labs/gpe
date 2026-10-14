gpu = False
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import gpe_functions_3D as gpe
import analysis_fns as plt_fns
import time
from scipy import fft as fftw
if gpu:
    import cupy as np
# import pyfftw.interfaces as fftw

# physical constants
hbar = 1.05e-34 # J.s
a0 = 5.29e-11 # m

# experimental parameters
wx = 2*np.pi*45
wy = 2*np.pi*45
wz = 2*np.pi*133
trap_ratio = wz/wx
add = 130*a0
ascat = 82*a0
Natoms = 15000
m = 164*1.66e-27
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
az = np.sqrt(hbar/(m*wz))
guess = [0.5,7]
def Evarz(sigmar, sigmaz, trap_ratio, N_atoms, az, add, ascat, gLHY, wz):
    def f(kappa):
        if isinstance(kappa, np.ndarray):
            out = np.zeros(np.shape(kappa))
            out[kappa**2==1] = 0
            out1 = (1+2*kappa**2)/(1-kappa**2)+3*kappa**2*np.atan(np.sqrt(kappa**2-1))/(kappa**2-1)**(3/2)
            out2 = (1+2*kappa**2)/(1-kappa**2)-3*kappa**2*np.atanh(np.sqrt(1-kappa**2))/(1-kappa**2)**(3/2)
            out[kappa**2>1] = out1[np.isnan(out1) != True]
            out[kappa**2<1] = out2[np.isnan(out2) != True]
            return out
        else: 
            if kappa**2 == 1:
                return 0
            if kappa**2 > 1:
                return (1+2*kappa**2)/(1-kappa**2)+3*kappa**2*np.atan(np.sqrt(kappa**2-1))/(kappa**2-1)**(3/2)
            return (1+2*kappa**2)/(1-kappa**2)-3*kappa**2*np.atanh(np.sqrt(1-kappa**2))/(1-kappa**2)**(3/2)
    return (2/sigmar**2+1/sigmaz**2)+(1/(8*(trap_ratio)**2)*sigmar**2+1/16*sigmaz**2)+(8*N_atoms*add)/(np.sqrt(2*np.pi)*az*sigmar**2*sigmaz)*(ascat/add-f(sigmar/sigmaz))+128*gLHY*(N_atoms/(az**3))**(3/2)/(25*np.sqrt(5)*np.pi**(9/4)*sigmar**3*sigmaz**(3/2)*hbar*wz)

Evarfn = lambda x: Evarz(x[0], x[1], trap_ratio, Natoms, az, add, ascat, gLHY, wz)
sol = minimize(Evarfn,x0=guess)
print(sol)
guess_var = 0.6
guess = [((1-guess_var)+2*guess_var*np.random.random())*sol.x[0], ((1-guess_var)+2*guess_var*np.random.random())*sol.x[1]]
sigmar, sigmaz = np.abs(np.array(guess)*az/l)
sigmarv, sigmazv = np.abs(sol.x)
Evar = sol.fun*wz/wx
sr = np.linspace(0.1, 100, 1000)
sz = np.linspace(0.1, 20, 100)
sr, sz = np.meshgrid(sr,sz)
Eplot = Evarz(sr, sz, trap_ratio, Natoms, az, add, ascat, gLHY, wz)
contour_levels = np.min(Eplot)+np.logspace(-2,2, 30) 
plt.contourf(sr,sz, Eplot, levels=contour_levels, cmap="cividis")
plt.xscale("log", base=2)
plt.colorbar(label='Evar')
plt.scatter(np.abs(sol.x[0]), np.abs(sol.x[1]), color='red', marker='x', s=20, label='minimum')
plt.scatter(np.abs(guess[0]), np.abs(guess[1]), color='green', marker='x', s=20, label='guess')
plt.legend()
plt.show()

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
mu_TF =(15*gamma/(8*2**(3/2)*np.pi)*np.sqrt(anisotropy[0]*anisotropy[1]*anisotropy[2]))**(2/5)
mu_TF_dip =(15*dipole_strength/(8*2**(3/2)*np.pi)*np.sqrt(anisotropy[0]*anisotropy[1]*anisotropy[2]))**(2/5)
R_TF = np.sqrt(2*mu_TF)
R_z_TF = R_TF/trap_ratio
print(R_TF, R_z_TF)
Lx = np.abs(2*R_TF)
Ly = np.abs(2*R_TF)
Lz = np.abs(5*R_TF)
print(Lx, Ly, Lz)

grid_boundaries = [-Lx/2, Lx/2, -Ly/2, Ly/2, -Lz/2, Lz/2]
time_step = 5e-2
iterations = 1000
tolerance = 1e-5

potential_fn = lambda psi, grid, kgrid: gpe.dipolar_potential_LHY_cutoff(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction,LHY_strength)
energy_fn = lambda psi, grid, kgrid: gpe.energy_functional_dipole_LHY_cutoff(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction,LHY_strength)
guess = gpe.Gaussian_ansatz(grid_shape, grid_boundaries, sigmar, sigmar, sigmaz,noise_level=0.1)
# guess = gpe.Gaussian_ansatz(grid_shape, grid_boundaries, 10, 10, 10, noise_level=0.1)
start_time = time.time()
grid, psi, max_iter_LHY, errs, mus, converged, Es = gpe.run_simulation_ite(grid_shape,grid_boundaries,potential_fn,time_step,iterations,guess,tolerance=tolerance,iter_check=50, energy_fn=energy_fn, save_name="stable_droplet/E_LHY")
end_time = time.time()
print("elapsed time:", end_time-start_time, "s")
print("converged:", converged)
print("energy:", np.real(Es[-1][-1]))
print("variational energy:", Evar)
psi_fft = fftw.fftshift(fftw.fftn(psi))
xmin, xmax, ymin, ymax, zmin, zmax = grid_boundaries
kx = 2*np.pi*fftw.fftshift(fftw.fftfreq(Nx)*Nx/(xmax-xmin))
ky =  2*np.pi*fftw.fftshift(fftw.fftfreq(Ny)*Ny/(ymax-ymin))
kz =  2*np.pi*fftw.fftshift(fftw.fftfreq(Nz)*Nz/(zmax-zmin))
kgrid = np.meshgrid(kx,ky,kz,sparse=True) 

# plot result
save_folder = "results/droplet_reproduction_as_82"
var_soln = gpe.Gaussian_ansatz(grid_shape, grid_boundaries,sigmarv, sigmarv, sigmazv, noise_level=0)
n_gauss = np.abs(var_soln)**2
n_sim = np.abs(psi)**2
n_peak = np.max(n_sim)*Natoms/l**3
print("Peak density:", n_peak)
plt_fns.plot_density_unnormalized_int(grid,n_sim, sum_axes=("z",), save_name="0_xy_density.png", save_folder=save_folder, data_labels=("n",), units=(l*1e6, "um"))
plt_fns.plot_density_unnormalized_int(kgrid, np.abs(psi_fft)**2, data_labels=("n_FFT",))
plt_fns.plot_density_unnormalized_int(grid,n_sim,sum_axes=("y",), save_name="0_xz_density.png", save_folder=save_folder, data_labels=("n",), units=(l*1e6, "um"))

plt_fns.plot_density_unnormalized_int(grid,[n_sim, n_gauss],sum_axes=("y","z"), data_labels=("GPE",  "Variational"), save_name="x_density.png", save_folder=save_folder, units=(l*1e6, "um"))
plt_fns.plot_density_unnormalized_int(grid,[n_sim, n_gauss],sum_axes=("x","y"), data_labels=("GPE", "Variational"), save_name="z_density.png", save_folder=save_folder, units=(l*1e6, "um"))

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