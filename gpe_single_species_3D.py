import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import gpe_functions_3D as gpe
import analysis_fns as plt_fns

# physical constants
hbar = 1.05e-34 # J.s
a0 = 5.29e-11 # m
m = 166*1.66e-27


def runner(omegas, Natoms, interacs, grid_sizes, lengths, iterations, tolerance, time_step):
    # unpacking
    wx, wy, wz = omegas
    ascat, epsdd, Q5edd = interacs
    Nx, Ny, Nz = grid_sizes
    Lx, Ly, Lz = lengths

    ratio = wx/wz
    g = 4*np.pi*hbar**2*ascat/m
    add = epsdd*ascat
    gdd = 4*np.pi*hbar**2*add/m
    # Q5edd = 1+3*epsdd**2/2
    # y = (1-epsdd)/(3*epsdd)+0j
    # Q5edd = (3*epsdd)**(5/2)/48*((8+26*y+33*y**2)*np.sqrt(1+y) + 15*y**3*np.log((1+np.sqrt(1+y))/np.sqrt(y)))
    # assert (np.real(Q5edd) > 10*np.imag(Q5edd))
    # Q5edd = np.real(Q5edd)
    # print(Q5edd)
    gLHY = (32/(3*np.sqrt(np.pi)))*g*ascat**(3/2)*Q5edd

    # dimensionless units
    l = np.sqrt(hbar/(m*wz))
    tau = 1/wz

    # from variational scaled parameters
    gamma = Natoms*g/(hbar*wz*l**3)
    dipole_strength = gdd*Natoms/(hbar*wz*l**3)
    LHY_strength = gLHY*Natoms**(3/2)/(hbar*wz*l**(9/2))
    anisotropy = [(wx/wz)**2, (wy/wz)**2, 1]
    dipole_direction = [0, 0, 1]

    grid_shape = [Nx, Ny, Nz]

    mu_TF = (15*gamma/(8*2**(3/2)*np.pi)*np.sqrt(anisotropy[0]*anisotropy[1]*anisotropy[2]))**(2/5)
    # mu_TF_dip = (15*dipole_strength/(8*2**(3/2)*np.pi)*np.sqrt(anisotropy[0]*anisotropy[1]*anisotropy[2]))**(2/5)
    # R_TF = np.sqrt(2*mu_TF)
    # R_z_TF = R_TF/ratio

    grid_boundaries = [-Lx/2, Lx/2, -Ly/2, Ly/2, -Lz/2, Lz/2]

    # 8 um radius after nondimensionalization
    wall = lambda grid: gpe.cylinder(grid, amp=1e3, radius=12, width=2)
    ext_pot = lambda grid: gpe.harmonic(grid, anisotropy) + wall(grid)
    int_pot = lambda psi, kgrid: gpe.contact(gamma, psi, LHY_strength) + gpe.ddi(psi, kgrid, dipole_strength, dipole_direction)
    potential = lambda psi, grid, kgrid: ext_pot(grid) + int_pot(psi, kgrid)
    # potential_fn = lambda psi, grid, kgrid: gpe.harmonic_potential(psi,grid,kgrid,gamma,anisotropy)
    # energy_fn = lambda psi, grid, kgrid: gpe.energy_functional_dipole_LHY(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction,LHY_strength)
    energy_fn = None

    guess = gpe.ansatz(grid_shape, grid_boundaries, ('u', 'u', 'h'), (wx/wz, wy/wz, 1), noise_level=0.1, wall=wall)
    grid, psi, max_iter_LHY, errs, mus, converged, Es = gpe.run_simulation_ite(grid_shape, grid_boundaries, potential, time_step, iterations, guess, 
                                                                            tolerance=tolerance, iter_check=100, energy_fn=energy_fn, save_name=None)
    print(f"Converged: {converged}")
    return grid, psi, ext_pot, guess, max_iter_LHY, errs


def plotter(grid, psi, ext_pot, guess, max_iter_LHY, errs, labels):
    external_potential = ext_pot(grid)
    plt_fns.plot_3D(grid, guess, external_potential, units=(l*1e6, "μm"), save_name="ansatz.png", save_folder="results/", show=True)
    # print("energy:", np.real(Es[-1][-1]))
    # psi_fft = fftw.fftshift(fftw.fftn(psi))
    # xmin, xmax, ymin, ymax, zmin, zmax = grid_boundaries
    # kx = 2*np.pi*fftw.fftshift(fftw.fftfreq(grid_sizes[0])*grid_sizes[0]/(xmax-xmin))
    # ky = 2*np.pi*fftw.fftshift(fftw.fftfreq(grid_sizes[1])*grid_sizes[1]/(ymax-ymin))
    # kz = 2*np.pi*fftw.fftshift(fftw.fftfreq(grid_sizes[2])*grid_sizes[2]/(zmax-zmin))
    # kgrid = np.meshgrid(kx, ky, kz, sparse=True) 

    # plot result
    n_sim = np.abs(psi)**2
    # n_TF = gpe.thomas_fermi_density(grid_shape, grid_boundaries, gamma, anisotropy)
    # n_dipolar_TF = gpe.thomas_fermi_cyl_dipole (grid_shape, grid_boundaries, epsdd, gamma,trap_ratio)

    name = rf"n : N={labels[0]:.0f}, $a_s$={labels[1]:g}$a_0$, $\epsilon_{{dd}}$={labels[2]}"
    saveas = f"N_{labels[0]:.0f}_as_{labels[1]:g}_edd_{labels[2]:g}"
    plt_fns.plot_density_unnormalized_int(grid, n_sim, sum_axes=("z",), save_name=saveas+"_plane.png", save_folder=save_folder, data_labels=("n",), units=(l*1e6, "μm"), title=name)
    # plt_fns.plot_density_unnormalized_int(grid, n_sim, sum_axes=("y",), save_name="0_xz_density.png", save_folder=save_folder, data_labels=("n",), units=(l*1e6, "um"))
    # plt_fns.plot_density_unnormalized_int(kgrid, np.abs(psi_fft)**2, sum_axes=("z",), data_labels=("n_FFT",))
    # plt_fns.plot_density_unnormalized_int(kgrid, np.abs(psi_fft)**2, sum_axes=("y",), data_labels=("n_FFT",))
    # plt_fns.plot_density_unnormalized_int(grid, [n_sim], sum_axes=("y", "z"), data_labels=("GPE",), save_name=saveas+"_cut.png", save_folder=save_folder, units=(l*1e6, "μm"), title=name)
    # plt_fns.plot_density_unnormalized_int(grid, [n_sim, n_TF], sum_axes=("x","y"), data_labels=("GPE","TF"), save_name="z_density.png", save_folder=save_folder, units=(l*1e6, "um"))
    plt_fns.plot_3D(grid, psi, ext_pot, units=(l*1e6, "μm"), save_name="output.png", save_folder="results/", show=True)

    # plot convergence
    iters_LHY = np.arange(0, max_iter_LHY+1, 1)
    plt.plot(iters_LHY, errs)
    plt.title("Error")
    plt.xlabel("Iteration")
    plt.yscale('log')
    plt.legend()
    plt.show()

    # plt.plot(iters_LHY, mus, label="LHY")
    # plt.title("Chemical Potential")
    # plt.legend()
    # plt.show()
    
    return grid, n_sim


# realistic experimental parameters
omegas = (2*np.pi*0, 2*np.pi*0, 2*np.pi*193)
Natoms = 2e4
# s wave scattering length is determined by magnetic field, which is related to epsilon!
# interacs = [(120*a0, 0.1, 0), (80*a0, 0.5, 0), (65*a0, 1.0, 0), (60*a0, 1.25, 0), (50*a0, 1.5, 0), (45*a0, 1.75, 0)] # (ascat, epsdd, Q5edd)
add = 66*a0
epsdd = 1.6
interacs = [(add/epsdd, epsdd, (1+3*epsdd**2/2))] # (ascat, epsdd=0.6 to 1.8, Q5edd)
# simulation grid
grid_sizes = (2**8, 2**8, 2**7)
lengths = (30, 30, 30)
its = 5000
tol = 3e-2 # go smaller (smooths unevenness)
step = 3e-3

# add noise at every step, first converge then add noise, seed with droplets (sine w/ and w/o noise)

l = np.sqrt(hbar/(m*omegas[2]))
save_folder = "results/"

for ints in interacs:
    grid, psi, ext_pot, guess, max_iter_LHY, errs = runner(omegas, Natoms, ints, grid_sizes, lengths, its, tol, step)
    labels = (Natoms, ints[0]/a0, ints[1])
    grid, n_sim = plotter(grid, psi, ext_pot, guess, max_iter_LHY, errs, labels)

tail = lambda x, A, x0, C: C + A/(x-x0)**1

coords = np.ravel(grid[0])*l*1e6
data = n_sim[int(grid_sizes[1]/2), :, int(grid_sizes[2]/2)]
mask = (coords > -6.5) & (coords < 0.0)
# popt, pcov = curve_fit(tail, coords[mask], data[mask], p0=(1e-3, -7.5, 6e-3), bounds=([0, -10, 0], [np.inf, 0, np.inf]))
# print(np.sqrt(np.diag(pcov)))
# n0lz = popt[0] * np.sqrt(2*np.pi) * (1 + 2*epsdd) / (3*epsdd)
# print(n0lz)

plt.plot(coords, data, label="GPE")
# plt.plot(coords[mask], tail(coords[mask], *popt), label=f"Fit: Power={1.00:.2f}")
plt.xlabel("x (μm)")
plt.ylabel("n")
plt.title(f"Density Profile: N={Natoms:g}, $a_s$={ints[0]/a0:g}$a_0$, $\epsilon_{{dd}}$={ints[1]:.2f}")
plt.legend()
plt.grid(which='both', alpha=0.25)
plt.savefig(save_folder + f"N_{Natoms:g}_as_{ints[0]/a0:g}_edd_{ints[1]:.2f}_cut.png")
plt.show()