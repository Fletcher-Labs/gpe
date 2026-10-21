gpu = True
import numpy as np
from matplotlib import pyplot as plt
import analysis_fns as plt_fns
from scipy.optimize import minimize_scalar
from scipy import fft as fftw
import os
from tqdm import tqdm

if gpu:
    import mlx.core as np


# functions
def run_simulation_ite(grid_shape, grid_boundaries, potential_fn, dt_input, iterations, guess, iter_check=100, plot_all=False, tolerance=1e-4, use_tolerance=True, save_name=None, energy_fn=None):
    grid, kgrid, psi = initialize_simulation(grid_shape, grid_boundaries, guess)
    errs = []
    mus = []
    converged = False
    max_iter = iterations-1
    Es = []
    labels = None
    dt = dt_input
    for i in tqdm(range(iterations)):
        if energy_fn:
            labels, E_components = energy_fn(psi, grid, kgrid)
            if i % iter_check == 0:
                Es.append(E_components)
                Emax = np.max(np.abs(np.array(list(E_components))))
                print(np.abs(Emax))
                dt = dt_input/np.abs(Emax)
                dt = float(dt)
                print("dt: ", dt)
        if i % iter_check == 0:
            if plot_all:
                plt_fns.plot_density_unnormalized_int(grid, np.abs(psi)**2)
                plt.show()
        psi_old = psi
        psi, psi_norm = imaginary_time_propagation_step(psi_old, grid, kgrid, potential_fn, dt)
        err = np.linalg.norm((psi - psi_old)/ dt) / np.linalg.norm(psi)
        # err = np.sum(np.abs(np.abs(psi)-np.abs(psi_old)))/np.sum(np.abs(psi))
        errs.append(err)
        if i % iter_check == 0:
            print(f"Iteration: {i} -- Error: {err:.3e} -- Norm: {psi_norm:.3e}")
        if err > 1e1:
            V = potential_fn(psi, grid, kgrid)
            print("peak:", float(np.max(np.abs(psi)**2)), "Vmin:", float(np.min(V.real)),"Vmax:", float(np.max(V.real)), "max exponent:", float(np.max((-dt * V).real)))
        if np.isnan(err):
            print("nan")
            converged = False
            max_iter=i
            break
        mu_check = -0.5*np.log(psi_norm)/dt
        mus.append(mu_check)
        if err < tolerance and use_tolerance:
            converged = True
            print("Converged!")
            max_iter = i
            break
    if energy_fn:
        Es = np.array(Es)
        for i, label in enumerate(labels):
            E_component = Es[:,i]
            plt.plot(E_component, label=label)
        plt.legend()
        if save_name:
            file_path = save_name + "E.png"
            directory, filename = os.path.split(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(file_path)
    if save_name:
        filename = save_name+".npz"
        # np.savez(filename, psi=psi, err=np.array(errs), mus=np.array(mus), Es=np.array(Es))
    return grid, psi, max_iter, errs, mus, converged, Es


def run_simulation_rte(grid_shape, grid_boundaries, potential_fn, dt, iterations, guess):
    grid,kgrid,psi = initialize_simulation(grid_shape, grid_boundaries, guess)
    for i in range(iterations):
        psi, _ = real_time_propagation_step(psi,grid,kgrid,potential_fn,dt)
    return grid, psi


def initialize_simulation(grid_shape, grid_boundaries, guess, plot_guess=False):
    xmin, xmax, ymin, ymax, zmin, zmax = np.array(grid_boundaries).flatten()
    x = np.linspace(xmin,xmax,num=grid_shape[0])
    y = np.linspace(ymin,ymax,num=grid_shape[1])
    z = np.linspace(zmin,zmax,num=grid_shape[2])
    Nx, Ny, Nz = grid_shape
    kx = 2*np.pi*np.fft.fftfreq(Nx)*Nx/(xmax-xmin)
    ky = 2*np.pi*np.fft.fftfreq(Ny)*Ny/(ymax-ymin)
    kz = 2*np.pi*np.fft.fftfreq(Nz)*Nz/(zmax-zmin)
    grid = np.meshgrid(x, y, z, sparse=True) 
    kgrid = np.meshgrid(kx, ky, kz, sparse=True) 
    psi = guess
    if plot_guess:
        plt_fns.plot_density_unnormalized_int(grid, np.abs(psi)**2)
        plt.show()
    return (grid, kgrid, psi)


def imaginary_time_propagation_step(psi, grid, kgrid, potential, dt):
    x, y, z = grid
    Nx, Ny, Nz = np.array(x).size, np.array(y).size, np.array(z).size
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    V = potential(psi, grid, kgrid)
    # dt = 1 / np.max(np.abs(V)) * dt if np.max(np.abs(V)) > 0 else dt
    psi = kinetic_propagator(psi, grid, kgrid, dt/2)
    psi = np.exp(-dt*V)*psi
    psi = kinetic_propagator(psi, grid, kgrid, dt/2)
    psi_norm = np.sum(np.abs(psi)**2*dLx*dLy*dLz)
    # print(dt)
    return psi/np.sqrt(psi_norm), psi_norm


def real_time_propagation_step(psi, grid, kgrid, potential_propagator, dt):
    return imaginary_time_propagation_step(psi, grid, kgrid, potential_propagator, 1j*dt)


def contact(gamma, psi, LHY_strength=0):
    return gamma*np.abs(psi)**2 + LHY_strength*np.abs(psi)**3


def harmonic(grid, anisotropy):
    x, y, z = grid
    return 1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)


def ddi(psi, kgrid, dipole_strength, dipole_direction):
    fft_n = fftw.fftn(np.abs(psi)**2)
    kx, ky, kz = kgrid
    kabs = np.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    dipole_fft = dipole_strength*(3*costheta**2-1)
    dipole_fft[kabs==0] = 0
    dipolar = fftw.ifftn(dipole_fft*fft_n)
    return dipolar


def box_2D(grid, amp, xlen, ylen, width=None):
    x, y, _ = (np.array(a) for a in grid)   
    inside = (np.abs(x) < xlen / 2) & (np.abs(y) < ylen / 2)
    if width:
        outside = (np.abs(x) < xlen / 2 + width) & (np.abs(y) < ylen / 2 + width)
        return np.where(outside & ~inside, amp, 0.0)
    return np.where(~inside, amp, 0.0)


def cylinder(grid, amp, radius, width=None):
    x, y, _ = (np.array(a) for a in grid)
    r = np.sqrt(x**2 + y**2)
    inside = r < radius
    if width:
        outside = r > radius + width
        return np.where(~outside & ~inside, 0.0, amp)
    return np.where(~inside, amp, 0.0)


def dipolar_potential_LHY(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction,LHY_strength):
    x, y, z = grid
    harmonic_plus_contact = 1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)+gamma*np.abs(psi)**2
    # dipolar term
    fft_n = fftw.fftn(np.abs(psi)**2)
    kx, ky, kz = kgrid
    kabs = np.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    dipole_fft = dipole_strength*(3*costheta**2-1)
    dipole_fft[kabs==0] = 0
    dipolar = fftw.ifftn(dipole_fft*fft_n)
    LHY = LHY_strength*np.abs(psi)**3
    return dipolar+harmonic_plus_contact+LHY


def dipolar_potential_LHY_cutoff(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction,LHY_strength):
    x, y, z = grid
    cutoff_radius = np.max([np.max(x), np.max(y), np.max(z)])
    harmonic_plus_contact = 1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)+gamma*np.abs(psi)**2
    # dipolar term
    fft_n = fftw.fftn(np.abs(psi)**2)
    kx, ky, kz = kgrid
    kabs = np.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    dipole_fft = (1+3*np.cos(cutoff_radius*kabs)/(cutoff_radius*kabs)**2-3*np.sin(cutoff_radius*kabs)/(cutoff_radius*kabs)**3)*dipole_strength*(3*costheta**2-1)
    dipole_fft[kabs==0] = 0
    dipolar = fftw.ifftn(dipole_fft*fft_n)
    LHY = LHY_strength*np.abs(psi)**3
    return dipolar+harmonic_plus_contact+LHY


def kinetic_propagator(psi, grid, kgrid, dt):
    fft_psi = np.fft.fftn(psi)
    kx, ky, kz = kgrid
    ksquared = kx**2+ky**2+kz**2
    return np.fft.ifftn(np.exp(-1/2*ksquared*dt)*fft_psi)


def _profile_1d(coord, kind, lo, hi, omega):
    if kind == "h":
        return (omega / np.pi) ** 0.25 * np.exp(-omega * coord**2 / 2)
    if kind == "u":
        L = hi - lo
        # return np.sqrt(2 / L)
        return np.sin(2 *np.pi * (coord - lo) / L)
    raise ValueError("trap type must be 'h' or 'u'")


def ansatz(grid_shape, grid_boundaries, trap_types=("h", "h", "h"), trap_frequencies=(1, 1, 1), noise_level=0.0, wall=None):
    if isinstance(trap_types, str):
        trap_types = (trap_types,) * 3
    xmin, xmax, ymin, ymax, zmin, zmax = grid_boundaries
    x = np.linspace(xmin, xmax, num=grid_shape[0])
    y = np.linspace(ymin, ymax, num=grid_shape[1])
    z = np.linspace(zmin, zmax, num=grid_shape[2])
    grid = np.meshgrid(x, y, z, sparse=True)  # array axes are (y, x, z), as in the rest of the code
    x, y, z = grid
    Nx, Ny, Nz = np.array(x).size, np.array(y).size, np.array(z).size
    Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
    bounds = ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    psi = 1.0
    for coord, kind, (lo, hi), omega in zip((x, y, z), trap_types, bounds, trap_frequencies):
        psi = psi * _profile_1d(coord, kind, lo, hi, omega)
    psi = psi * (1 + noise_level * np.random.uniform(low=-1.0, high=1.0, shape=psi.shape))
    psi = psi * np.exp(2 * np.pi * 1j * noise_level * np.random.uniform(low=-1.0, high=1.0, shape=psi.shape))
    if wall is not None:                       
        psi = psi * (np.array(wall(grid)) == 0)
    # renormalize on the grid (same convention as Gaussian_ansatz)
    dLx, dLy, dLz = Lx / (Nx - 1), Ly / (Ny - 1), Lz / (Nz - 1)
    psi_norm = np.sum(np.abs(psi) ** 2 * dLx * dLy * dLz)
    return psi / np.sqrt(psi_norm)


def Gaussian_ansatz(grid_shape, grid_boundaries, sigmax, sigmay, sigmaz, noise_level=0):
    xmin, xmax, ymin, ymax, zmin, zmax = grid_boundaries
    x = np.linspace(xmin, xmax, num=grid_shape[0])
    y = np.linspace(ymin, ymax, num=grid_shape[1])
    z = np.linspace(zmin, zmax, num=grid_shape[2])
    grid = np.meshgrid(x, y, z, sparse=True)
    x, y, z = grid
    Nx, Ny, Nz = np.array(x).size, np.array(y).size, np.array(z).size
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    # rng = np.random.default_rng()
    psi = np.exp(-2*((x**2)/(sigmax**2)+y**2/sigmay**2 + z**2/sigmaz**2))
    # noise = -1+2*(rng.random(np.shape(psi)))
    # psi = psi + psi*noise*noise_level
    # random_phase = 2*np.pi*1j*noise_level*(rng.random(np.shape(psi)))
    # psi = psi * np.exp(random_phase)    
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    psi_norm = np.sum(np.abs(psi)**2*dLx*dLy*dLz)
    return psi/np.sqrt(psi_norm)


def Gaussian_ansatz_TF(grid_shape, grid_boundaries, gamma, anisotropy):
    xmin, xmax, ymin, ymax, zmin, zmax = np.array(grid_boundaries).flatten()
    x = np.linspace(xmin, xmax, num=grid_shape[0])
    y = np.linspace(ymin, ymax, num=grid_shape[1])
    z = np.linspace(zmin, zmax, num=grid_shape[2])
    grid = np.meshgrid(x, y, z, sparse=True) 
    x, y, z = grid
    Nx, Ny, Nz = np.array(x).size, np.array(y).size, np.array(z).size
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    mu = (15*gamma/(8*2**(3/2)*np.pi)*np.sqrt(anisotropy[0]*anisotropy[1]*anisotropy[2]))**(2/5)
    Rx, Ry, Rz = np.sqrt(2*mu/(anisotropy))
    psi = np.exp(-2*(x**2/Rx**2+y**2/Ry**2+z**2/Rz**2))
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    psi_norm = np.sum(np.abs(psi)**2*dLx*dLy*dLz)
    return psi/np.sqrt(psi_norm)


def thomas_fermi_density(grid_shape, grid_boundaries, gamma, anisotropy):
    xmin, xmax, ymin, ymax, zmin, zmax = np.array(grid_boundaries).flatten()
    x = np.linspace(xmin,xmax,num=grid_shape[0])
    y = np.linspace(ymin,ymax,num=grid_shape[1])
    z = np.linspace(zmin,zmax,num=grid_shape[2])
    Nx, Ny, Nz = grid_shape
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    x, y, z = np.meshgrid(x,y,z,sparse=True) 
    potential = 1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)
    mu = (15*gamma/(8*2**(3/2)*np.pi)*np.sqrt(anisotropy[0]*anisotropy[1]*anisotropy[2]))**(2/5)
    tf = (mu-potential)/gamma
    tf[tf<0] = 0
    tf_norm = np.sum(np.array(tf*dLx*dLy*dLz))
    return tf/tf_norm


def thomas_fermi_cyl_dipole(grid_shape, grid_boundaries, epsdd, gamma,trap_ratio,return_mu=False):
    def fdip(kappa):
        if kappa**2 == 1:
            return 0
        if kappa**2 > 1:
            return (1+2*kappa**2)/(1-kappa**2)+3*kappa**2*np.atan(np.sqrt(kappa**2-1))/(kappa**2-1)**(3/2)
        return (1+2*kappa**2)/(1-kappa**2)-3*kappa**2*np.atanh(np.sqrt(1-kappa**2))/(1-kappa**2)**(3/2)
    def aspect_ratio(edd, trap_ratio):
        fn = lambda kappa: 3*kappa**2*edd*((trap_ratio**2/2+1)*fdip(kappa)/(1-kappa**2)-1)+(edd-1)*(kappa**2-trap_ratio**2) 
        fn2 = lambda kappa: fn(kappa)**2
        sol = minimize_scalar(fn2,bracket=(0.1,2),bounds=(0,50),method="bounded",tol=1e-9)
        return sol.x
    # calculate TF radii
    kappa = aspect_ratio(epsdd, trap_ratio)
    if kappa == 1:
        R_rho = (15/(4*np.pi)*kappa*gamma*(1+epsdd*(3/2*kappa**2*(-2/5) - 1)))**(1/5)
    else:
        R_rho = (15/(4*np.pi)*kappa*gamma*(1+epsdd*(3/2*kappa**2*fdip(kappa)/(1-kappa**2) - 1)))**(1/5) 
    R_z = R_rho/kappa 
    # calculate density profile
    xmin, xmax, ymin, ymax, zmin, zmax = np.ravel(grid_boundaries)
    x = np.linspace(xmin,xmax,num=grid_shape[0])
    y =  np.linspace(ymin,ymax,num=grid_shape[1])
    z =  np.linspace(zmin,zmax,num=grid_shape[2])
    Nx, Ny, Nz = grid_shape
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y),  np.max(z)-np.min(z)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    x,y,z = np.meshgrid(x,y,z,sparse=True) 
    tf = (1-(x**2+y**2)/R_rho**2-z**2/R_z**2)
    tf[tf<0] = 0
    tf_norm = np.sum(tf*dLx*dLy*dLz)
    if return_mu:
        mu = 15/(8*np.pi*R_rho**2*R_z)*gamma*(1-epsdd*fdip(kappa))
        return mu
    return tf/tf_norm


def energy_functional_dipole_LHY(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction,LHY_strength):
    fft_psi = fftw.fftn(psi)
    fft_n = fftw.fftn(np.abs(psi)**2)
    kx, ky, kz = kgrid
    x, y, z = grid
    Nx, Ny, Nz = np.array(x).size, np.array(y).size, np.array(z).size
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    kabs = np.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    costheta = np.nan_to_num(costheta)
    dipole_fft = dipole_strength*(3*costheta**2-1)
    dipole_fft[kabs==0] = 0
    KE = np.sum(np.conj(psi)*fftw.ifftn(kabs**2/2*fft_psi)*dLx*dLy*dLz)
    harmonic = np.sum(1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)*np.abs(psi)**2*dLx*dLy*dLz)
    contact = np.sum(1/2*gamma*np.abs(psi)**4*dLx*dLy*dLz)
    LHY = np.sum(2/5*LHY_strength*np.abs(psi)**5*dLx*dLy*dLz)
    dipolar = np.sum(1/2*fftw.ifftn(dipole_fft*fft_n)*np.abs(psi)**2*dLx*dLy*dLz)
    E = KE+harmonic+contact+LHY+dipolar
    # print("KE: ", KE)
    # print("harmonic: ", harmonic)
    # print("contact: ", contact)
    # print("dipolar: ", dipolar)
    # print("LHY: ", LHY)
    # print("E: ", E)
    labels = ("KE", "harmonic", "contact", "dipolar", "LHY", "E")
    energies = (KE, harmonic, contact, dipolar, LHY, E)
    return labels, energies


def energy_functional_dipole_LHY_cutoff(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction,LHY_strength):
    fft_psi = fftw.fftn(psi)
    fft_n = fftw.fftn(np.abs(psi)**2)
    kx, ky, kz = kgrid
    x, y, z = grid
    cutoff_radius = np.max([np.max(x), np.max(y), np.max(z)])
    Nx, Ny, Nz = np.array(x).size, np.array(y).size, np.array(z).size
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    kabs = np.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    costheta = np.nan_to_num(costheta)
    dipole_fft = (1+3*np.cos(cutoff_radius*kabs)/(cutoff_radius*kabs)**2-3*np.sin(cutoff_radius*kabs)/(cutoff_radius*kabs)**3)*dipole_strength*(3*costheta**2-1)
    dipole_fft[kabs==0] = 0
    KE = np.sum(np.conj(psi)*fftw.ifftn(kabs**2/2*fft_psi)*dLx*dLy*dLz)
    harmonic = np.sum(1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)*np.abs(psi)**2*dLx*dLy*dLz)
    contact = np.sum(1/2*gamma*np.abs(psi)**4*dLx*dLy*dLz)
    LHY = np.sum(2/5*LHY_strength*np.abs(psi)**5*dLx*dLy*dLz)
    dipolar = np.sum(1/2*fftw.ifftn(dipole_fft*fft_n)*np.abs(psi)**2*dLx*dLy*dLz)
    E = KE+harmonic+contact+LHY+dipolar
    # print("KE: ", KE)
    # print("harmonic: ", harmonic)
    # print("contact: ", contact)
    # print("dipolar: ", dipolar)
    # print("LHY: ", LHY)
    # print("E: ", E)
    labels = ("KE", "harmonic", "contact", "dipolar", "LHY", "E")
    energies = (KE, harmonic, contact, dipolar, LHY, E)
    return labels, energies


def total_energy_functional_dipole_LHY(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction,LHY_strength):
    fft_psi = fftw.fftn(psi)
    fft_n = fftw.fftn(np.abs(psi)**2)
    kx, ky, kz = kgrid
    x, y, z = grid
    Nx, Ny, Nz = np.array(x).size, np.array(y).size, np.array(z).size
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    kabs = np.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    costheta = np.nan_to_num(costheta)
    dipole_fft = dipole_strength*(3*costheta**2-1)
    dipole_fft[kabs==0] = 0
    KE = np.sum(np.conj(psi)*fftw.ifftn(kabs**2/2*fft_psi)*dLx*dLy*dLz)
    harmonic = np.sum(1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)*np.abs(psi)**2*dLx*dLy*dLz)
    contact = np.sum(1/2*gamma*np.abs(psi)**4*dLx*dLy*dLz)
    LHY = np.sum(2/5*LHY_strength*np.abs(psi)**5*dLx*dLy*dLz)
    dipolar = np.sum(1/2*fftw.ifftn(dipole_fft*fft_n)*np.abs(psi)**2*dLx*dLy*dLz)
    E = KE+harmonic+contact+LHY+dipolar
    return E


def total_energy_functional_dipole_LHY_cutoff(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction,LHY_strength):
    fft_psi = fftw.fftn(psi)
    fft_n = fftw.fftn(np.abs(psi)**2)
    kx, ky, kz = kgrid
    x, y, z = grid
    cutoff_radius = np.max([np.max(x), np.max(y), np.max(z)])
    Nx, Ny, Nz = np.array(x).size, np.array(y).size, np.array(z).size
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    kabs = np.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    costheta = np.nan_to_num(costheta)
    dipole_fft = (1+3*np.cos(cutoff_radius*kabs)/(cutoff_radius*kabs)**2-3*np.sin(cutoff_radius*kabs)/(cutoff_radius*kabs)**3)*dipole_strength*(3*costheta**2-1)
    dipole_fft[kabs==0] = 0
    KE = np.sum(np.conj(psi)*fftw.ifftn(kabs**2/2*fft_psi)*dLx*dLy*dLz)
    harmonic = np.sum(1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)*np.abs(psi)**2*dLx*dLy*dLz)
    contact = np.sum(1/2*gamma*np.abs(psi)**4*dLx*dLy*dLz)
    LHY = np.sum(2/5*LHY_strength*np.abs(psi)**5*dLx*dLy*dLz)
    dipolar = np.sum(1/2*fftw.ifftn(dipole_fft*fft_n)*np.abs(psi)**2*dLx*dLy*dLz)
    E = KE+harmonic+contact+LHY+dipolar
    return E