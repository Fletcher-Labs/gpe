import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
from cupy import fft as fftw
import cupy as cp
from plot_fns_gpu import plot_density_unnormalized_int
import os

# functions
def run_simulation_ite(grid_shape, grid_boundaries, potential_fn, dt_input, iterations, guess, iter_check=100, tolerance=1e-4,use_tolerance=True, save_name=None, print_all=True, energy_fn=None, animation=None):
    grid,kgrid,psi = initialize_simulation(grid_shape, grid_boundaries, guess)
    errs = []
    mus = []
    max_iter = iterations-1
    Es = []
    artists1 = []
    artists2 = []
    labels = None
    converged = False
    for i in range(iterations):
        if energy_fn:
            labels, E_components = energy_fn(psi, grid, kgrid)
            Es.append(E_components)
            Emax = max(cp.abs(cp.array(E_components)))
            dt = dt_input/cp.abs(Emax)
            dt = float(dt)
        psi_old = psi
        psi, psi_norm = imaginary_time_propagation_step(psi_old,grid,kgrid,potential_fn,dt)
        err = cp.sum(cp.abs(cp.abs(psi)-cp.abs(psi_old)))/(cp.sum(cp.abs(psi)))
        errs.append(float(err.get()))
        # if len(Es) > 2:
        #     deltaE = cp.abs(cp.real((cp.array(list(Es[-1]))-cp.array(list(Es[-2])))))
        #     err = cp.max(deltaE)/cp.abs(Es[-1][int(cp.argmax(deltaE))])
        #     err = float(err)
        mu_check = -1/2*np.log(psi_norm)/dt
        if i % iter_check == 0:
            print("iteration: ", i)
            print("err =", err)
            print("dt: ", dt)
        mus.append(float(mu_check.get()))
        if err < tolerance and use_tolerance:
            converged = True
            print("converged!")
            max_iter = i
            break
    if energy_fn:
        Es = cp.array(Es)
        for i, label in enumerate(labels):
            E_component = Es[:,i]
            E_component = E_component.get()
            plt.plot(E_component, label=label)
        plt.legend()
        if save_name:
            file_path = save_name+"E.png"
            directory, filename = os.path.split(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(file_path)
    if save_name:
        filename = save_name+".cpz"
        cp.savez(filename, psi=psi, err=cp.array(errs))
    return grid, psi, max_iter, errs, mus, converged, Es

def initialize_simulation(grid_shape, grid_boundaries, guess, plot_guess=False):
    xmin, xmax, ymin, ymax, zmin, zmax = grid_boundaries
    x = cp.linspace(xmin,xmax,num=grid_shape[0])
    y =  cp.linspace(ymin,ymax,num=grid_shape[1])
    z =  cp.linspace(zmin,zmax,num=grid_shape[2])
    Nx, Ny, Nz = grid_shape
    kx = 2*cp.pi*cp.fft.fftfreq(Nx)*Nx/(xmax-xmin)
    ky =  2*cp.pi*cp.fft.fftfreq(Ny)*Ny/(ymax-ymin)
    kz =  2*cp.pi*cp.fft.fftfreq(Nz)*Nz/(zmax-zmin)
    grid = cp.meshgrid(x,y,z,sparse=True)
    kgrid = cp.meshgrid(kx,ky,kz,sparse=True)
    psi = guess
    if plot_guess:
        print("Guess:")
        plot_density_unnormalized_int(grid,(cp.abs(psi)**2).get())
        plt.show()
    print("simulation initialized")
    return (grid,kgrid,psi)

def imaginary_time_propagation_step(psi, grid, kgrid, potential, dt):
    x, y, z = grid
    Nx, Ny, Nz = cp.size(x), cp.size(y), cp.size(z)
    Lx, Ly, Lz = cp.max(x)-cp.min(x), cp.max(y)-cp.min(y), cp.max(z)-cp.min(z)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    psi = kinetic_propagator(psi, grid, kgrid, dt/2)
    psi = cp.exp(-dt*potential(psi, grid, kgrid))*psi
    psi = kinetic_propagator(psi, grid, kgrid, dt/2)
    psi_norm = cp.sum(cp.abs(psi)**2*dLx*dLy*dLz)
    while cp.isinf(psi_norm):
      psi = psi/10**10
      psi_norm =  cp.sum(cp.abs(psi)**2*dLx*dLy*dLz)
    return psi/cp.sqrt(psi_norm), psi_norm

def real_time_propagation_step(psi, grid, kgrid, potential_propagator, dt):
    return imaginary_time_propagation_step(psi, grid, kgrid, potential_propagator, 1j*dt)

def harmonic_potential(psi,grid,kgrid,gamma,anisotropy):
    x, y, z = grid
    return 1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)+gamma*cp.abs(psi)**2

def dipolar_potential(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction):
    x, y, z = grid
    harmonic_plus_contact = 1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)+gamma*cp.abs(psi)**2
    # dipolar term
    fft_n = fftw.fftn(cp.abs(psi)**2)
    kx, ky, kz = kgrid
    kabs = cp.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    dipole_fft = dipole_strength*(3*costheta**2-1)
    dipole_fft[kabs==0] = 0
    dipolar = fftw.ifftn(dipole_fft*fft_n)
    return dipolar+harmonic_plus_contact
    
def dipolar_potential_LHY(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction,LHY_strength):
    x, y, z = grid
    harmonic_plus_contact = 1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)+gamma*np.abs(psi)**2
    # dipolar term
    fft_n = fftw.fftn(np.abs(psi)**2)
    kx, ky, kz = kgrid
    kabs = cp.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    dipole_fft = dipole_strength*(3*costheta**2-1)
    dipole_fft[kabs==0] = 0
    dipolar = fftw.ifftn(dipole_fft*fft_n)
    LHY = LHY_strength*np.abs(psi)**3
    return dipolar+harmonic_plus_contact+LHY

def dipolar_potential_LHY_cutoff(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction,LHY_strength,cutoff_radius=None):
    x, y, z = grid
    if not cutoff_radius:
      cutoff_radius = cp.max(cp.array([cp.max(x), cp.max(y), cp.max(z)]))
    harmonic_plus_contact = 1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)+gamma*cp.abs(psi)**2
    # dipolar term
    fft_n = fftw.fftn(np.abs(psi)**2)
    kx, ky, kz = kgrid
    kabs = cp.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    dipole_fft = (1+3*cp.cos(cutoff_radius*kabs)/(cutoff_radius*kabs)**2-3*cp.sin(cutoff_radius*kabs)/(cutoff_radius*kabs)**3)*dipole_strength*(3*costheta**2-1)
    dipole_fft[kabs==0] = 0
    dipolar = fftw.ifftn(dipole_fft*fft_n)
    LHY = LHY_strength*np.abs(psi)**3
    return dipolar+harmonic_plus_contact+LHY

def cylindrical_dipolar_potential(psi,grid,kgrid,gamma,trap_ratio,dipole_strength):
    x, y, z = grid
    # harmonic potential
    harmonic_plus_contact = 1/2*(x**2+y**2+trap_ratio**2*z**2)+gamma*cp.abs(psi)**2
    # dipolar potential
    fft_n = fftw.fftn(cp.abs(psi)**2)
    kx, ky, kz = kgrid
    kabs = cp.sqrt(kx**2+ky**2+kz**2)
    costheta = (kz)/kabs
    costheta = cp.nan_to_num(costheta)
    dipole_fft = dipole_strength*(3*costheta**2-1)
    dipole_fft[kabs==0] = 0
    dipolar = fftw.ifftn(dipole_fft*fft_n)
    return dipolar+harmonic_plus_contact



def kinetic_propagator(psi, grid, kgrid, dt):
    fft_psi = cp.fft.fftn(psi)
    kx, ky, kz = kgrid
    ksquared = kx**2+ky**2+kz**2
    return cp.fft.ifftn(cp.exp(-1/2*ksquared*dt)*fft_psi)


def Gaussian_ansatz(grid_shape, grid_boundaries, sigmax, sigmay, sigmaz, noise_level):
    xmin, xmax, ymin, ymax, zmin, zmax = grid_boundaries
    x = cp.linspace(xmin,xmax,num=grid_shape[0])
    y =  cp.linspace(ymin,ymax,num=grid_shape[1])
    z =  cp.linspace(zmin,zmax,num=grid_shape[2])
    grid = cp.meshgrid(x,y,z,sparse=True)
    x, y, z = grid
    Nx, Ny, Nz = cp.size(x), cp.size(y), cp.size(z)
    Lx, Ly, Lz = cp.max(x)-cp.min(x), cp.max(y)-cp.min(y), cp.max(z)-cp.min(z)
    rng = cp.random.default_rng()
    psi = cp.exp(-2*((x**2)/(sigmax**2)+y**2/sigmay**2 + z**2/sigmaz**2))
    noise = -1+2*(rng.random(cp.shape(psi)))
    psi = psi + psi*noise*noise_level
    random_phase = 2*np.pi*1j*noise_level*(rng.random(cp.shape(psi)))
    psi = psi*cp.exp(random_phase)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    psi_norm = cp.sum(cp.abs(psi)**2*dLx*dLy*dLz)
    return psi/cp.sqrt(psi_norm)
    
def Gaussian_ansatz_three(grid_shape, grid_boundaries, sigmax, sigmay, sigmaz, d, noise_level):
    xmin, xmax, ymin, ymax, zmin, zmax = grid_boundaries
    x = cp.linspace(xmin,xmax,num=grid_shape[0])
    y =  cp.linspace(ymin,ymax,num=grid_shape[1])
    z =  cp.linspace(zmin,zmax,num=grid_shape[2])
    grid = cp.meshgrid(x,y,z,sparse=True)
    x, y, z = grid
    Nx, Ny, Nz = cp.size(x), cp.size(y), cp.size(z)
    Lx, Ly, Lz = cp.max(x)-cp.min(x), cp.max(y)-cp.min(y), cp.max(z)-cp.min(z)
    rng = cp.random.default_rng()
    psi = cp.exp(-2*((x**2)/(sigmax**2)+y**2/sigmay**2 + z**2/sigmaz**2))+cp.exp(-2*(((x+d)**2)/(sigmax**2)+y**2/sigmay**2 + z**2/sigmaz**2))+cp.exp(-2*(((x-d)**2)/(sigmax**2)+y**2/sigmay**2 + z**2/sigmaz**2))
    noise = -1+2*(rng.random(cp.shape(psi)))
    psi = psi + psi*noise*noise_level
    random_phase = 2*np.pi*1j*noise_level*(rng.random(cp.shape(psi)))
    psi = psi*cp.exp(random_phase)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    psi_norm = cp.sum(cp.abs(psi)**2*dLx*dLy*dLz)
    return psi/cp.sqrt(psi_norm)


def Gaussian_ansatz_TF(grid_shape, grid_boundaries, gamma,anisotropy):
    xmin, xmax, ymin, ymax, zmin, zmax = grid_boundaries
    x = cp.linspace(xmin,xmax,num=grid_shape[0])
    y =  cp.linspace(ymin,ymax,num=grid_shape[1])
    z =  cp.linspace(zmin,zmax,num=grid_shape[2])
    grid = cp.meshgrid(x,y,z,sparse=True)
    x, y, z = grid
    Nx, Ny, Nz = cp.size(x), cp.size(y), cp.size(z)
    Lx, Ly, Lz = cp.max(x)-cp.min(x), cp.max(y)-cp.min(y), cp.max(z)-cp.min(z)
    mu = (15*gamma/(8*2**(3/2)*cp.pi)*cp.sqrt(anisotropy[0]*anisotropy[1]*anisotropy[2]))**(2/5)
    Rx, Ry, Rz = cp.sqrt(2*mu/anisotropy[0]), cp.sqrt(2*mu/anisotropy[1]), cp.sqrt(2*mu/anisotropy[2])
    psi = cp.exp(-2*(x**2/Rx**2+y**2/Ry**2+z**2/Rz**2))
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    psi_norm = cp.sum(cp.abs(psi)**2*dLx*dLy*dLz)
    return psi/cp.sqrt(psi_norm)



def thomas_fermi_density(grid_shape, grid_boundaries, gamma,anisotropy):
    xmin, xmax, ymin, ymax, zmin, zmax = grid_boundaries
    x = cp.linspace(xmin,xmax,num=grid_shape[0])
    y =  cp.linspace(ymin,ymax,num=grid_shape[1])
    z =  cp.linspace(zmin,zmax,num=grid_shape[2])
    Nx, Ny, Nz = grid_shape
    Lx, Ly, Lz = cp.max(x)-cp.min(x), cp.max(y)-cp.min(y),  cp.max(z)-cp.min(z)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    x,y,z = cp.meshgrid(x,y,z,sparse=True)
    potential = 1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)
    mu = (15*gamma/(8*2**(3/2)*cp.pi)*cp.sqrt(anisotropy[0]*anisotropy[1]*anisotropy[2]))**(2/5)
    tf = (mu-potential)/gamma
    tf[tf<0] = 0
    tf_norm = cp.sum(tf*dLx*dLy*dLz)
    return tf/tf_norm

def thomas_fermi_cyl_dipole (grid_shape, grid_boundaries, epsdd, gamma,trap_ratio,return_mu=False,return_R=False):
    def fdip(kappa):
        if kappa**2 == 1:
            return 0
        if kappa**2 > 1:
            return (1+2*kappa**2)/(1-kappa**2)+3*kappa**2*np.atan(cp.sqrt(kappa**2-1))/(kappa**2-1)**(3/2)
        return (1+2*kappa**2)/(1-kappa**2)-3*kappa**2*np.atanh(cp.sqrt(1-kappa**2))/(1-kappa**2)**(3/2)
    def aspect_ratio(edd, trap_ratio):
        fn = lambda kappa: cp.asnumpy(3*kappa**2*edd*((trap_ratio**2/2+1)*fdip(kappa)/(1-kappa**2)-1)+(edd-1)*(kappa**2-trap_ratio**2))
        fn2 = lambda kappa: cp.asnumpy(fn(kappa)**2)
        sol = minimize_scalar(fn2,bracket=(0.1,2),bounds=(0,50),method="bounded",tol=1e-9)
        return sol.x
    # calculate TF radii
    kappa = aspect_ratio(float(epsdd.get()), float(trap_ratio.get()))
    if kappa == 1:
        R_rho = (15/(4*cp.pi)*kappa*gamma*(1+epsdd*(3/2*kappa**2*(-2/5) - 1)))**(1/5)
    else:
        R_rho = (15/(4*cp.pi)*kappa*gamma*(1+epsdd*(3/2*kappa**2*fdip(kappa)/(1-kappa**2) - 1)))**(1/5)
    R_z = R_rho/kappa
    # calculate density profile
    xmin, xmax, ymin, ymax, zmin, zmax = grid_boundaries
    x = cp.linspace(xmin,xmax,num=grid_shape[0])
    y =  cp.linspace(ymin,ymax,num=grid_shape[1])
    z =  cp.linspace(zmin,zmax,num=grid_shape[2])
    Nx, Ny, Nz = grid_shape
    Lx, Ly, Lz = cp.max(x)-cp.min(x), cp.max(y)-cp.min(y),  cp.max(z)-cp.min(z)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    x,y,z = cp.meshgrid(x,y,z,sparse=True)
    tf = (1-(x**2+y**2)/R_rho**2-z**2/R_z**2)
    tf[tf<0] = 0
    tf_norm = cp.sum(tf*dLx*dLy*dLz)
    if return_mu:
        mu = 15/(8*cp.pi*R_rho**2*R_z)*gamma*(1-epsdd*fdip(kappa))
        return mu
    if return_R:
        return R_rho, R_z
    return tf/tf_norm

def energy_functional_dipole_LHY(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction,LHY_strength):
    fft_psi = fftw.fftn(psi)
    fft_n = fftw.fftn(np.abs(psi)**2)
    kx, ky, kz = kgrid
    x, y, z = grid
    Nx, Ny, Nz = cp.size(x), cp.size(y), cp.size(z)
    Lx, Ly, Lz = cp.max(x)-cp.min(x), cp.max(y)-cp.min(y), cp.max(z)-cp.min(z)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    kabs = cp.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    costheta = cp.nan_to_num(costheta)
    dipole_fft = dipole_strength*(3*costheta**2-1)
    dipole_fft[kabs==0] = 0
    KE = cp.sum(cp.conj(psi)*fftw.ifftn(kabs**2/2*fft_psi)*dLx*dLy*dLz)
    harmonic = cp.sum(1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)*cp.abs(psi)**2*dLx*dLy*dLz)
    contact = cp.sum(1/2*gamma*cp.abs(psi)**4*dLx*dLy*dLz)
    LHY = cp.sum(2/5*LHY_strength*cp.abs(psi)**5*dLx*dLy*dLz)
    dipolar = cp.sum(1/2*fftw.ifftn(dipole_fft*fft_n)*cp.abs(psi)**2*dLx*dLy*dLz)
    E = KE+harmonic+contact+LHY+dipolar
    labels = ("KE", "harmonic", "contact", "dipolar", "LHY", "E")
    energies = (KE, harmonic, contact, dipolar, LHY, E)
    return labels, energies
