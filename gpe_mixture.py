import numpy as np
from matplotlib import pyplot as plt
import analysis_fns as plt_fns
from scipy.optimize import minimize_scalar
from scipy import fft as fftw
import scipy
import os
from matplotlib.animation import FFMpegWriter
from contextlib import nullcontext
import time 
from perlin_numpy import generate_perlin_noise_3d
import random

# functions
def run_simulation_ite(grid_shape, grid_boundaries, potential_propagator, kinetic_propagator, time_step, iterations, guess1, guess2, iter_check=100, tolerance=1e-4,use_tolerance=True, save_name=None, energy_fn=None, noise_iters=0, noise_level=0.3, animation=None, vm=10, perlin=False, save_data=False):
    """ Runs imaginary time evolution
    grid_shape: [Nx, Ny, Nz]
    grid_boundaries: [xmin, xmax, ymin, ymax, zmin, zmax]
    potential_propagator: for H = V+K, function with args (psi, grid, kgrid, dt) which applies e^(-V dt)
    kinetic_propagator: for H = V+K, function with args (psi, grid, kgrid, dt) which applies e^(-K dt) 
    time_step: time step, either in units of 1/E_max, if energy_fn is specified, or in units set by form of GP equation
    iterations: max number of iterations to run
    guess1: initial guess for psi1
    guess2: initial guess for psi2
    iter_check: increment with which to print simulation output
    tolerance: sets the convergence criterion
    use_tolerance: if False, ignores convergence criterion
    save_name: name with which to save data and energy plots
    energy_fn: specifies energy functional E[psi]
    noise_iters: number of iterations during which noise is added
    noise_level: percent noise to add to wavefunction during noise_iters
    animation: if None, no animation is saved. otherwise, an integer specifying increment over which to save frames to animation
    vm: sets maximum colorbar scale for animation
    perlin: boolean specifying whether to add perlin noise or random noise during noise_iters
    save_data: boolean specifying whether or not to save psi to a npz file
    """
    grid,kgrid,psi = initialize_simulation(grid_shape, grid_boundaries, guess1, guess2)
    errs = []
    mu1s = []
    mu2s = []
    Es = []
    converged = False
    max_iter=iterations-1
    # animation setup 
    im1 = title1 = writer1 = None
    im2 = title2 = writer2 = None
    writer1_context = writer2_context = nullcontext()
    if animation is not None and save_name:
        ani1, ani2 =  set_up_animation(save_name,psi,vm)
        im1, title1, writer1, writer1_context = ani1
        im2, title2, writer2, writer2_context = ani2
    dt = time_step
    with writer1_context, writer2_context:
        err = 1
        for i in range(iterations):
            # grab frames for animation
            if animation is not None and i % animation == 0:
                n1 = np.sum(np.abs(psi[0,:,:,:])**2,axis=2)
                n2 = np.sum(np.abs(psi[1,:,:,:])**2,axis=2)
                im1.set_data(n1)
                title1.set_text(f"Iteration {i}")
                writer1.grab_frame()
                im2.set_data(n2)
                title2.set_text(f"Iteration {i}")
                writer2.grab_frame()
            # calculate energies and update times step
            if energy_fn:
                labels, E_components = energy_fn(psi, grid, kgrid)
                Es.append(E_components)   
                Emax = max(np.abs(np.array(E_components)))
                dt = float(time_step/(np.abs(Emax)))
                # if len(Es) > 1:
                #     err = np.abs(Es[-1][-1]-Es[-2][-1])/np.abs(Es[-1][-1])
            psi_old = psi
            # actual propagation step 
            psi, psi_norm = imaginary_time_propagation_step(psi,grid,kgrid,potential_propagator,kinetic_propagator,dt)
            # if energy_fn is None:
            # calculation of error and chemical potential
            err = np.sum(np.abs(np.abs(psi)-np.abs(psi_old)))/np.sum(np.abs(psi))
            mu1, mu2 = -np.log(psi_norm)/dt
            mu1s.append(float(mu1))
            mu2s.append(float(mu2))
            errs.append(float(err))
            if i % iter_check == 0: 
                print("dt:", dt)
                print("iteration:", i)
                if energy_fn:
                    print(labels, Es[-1])
                # print(f"$\Delta E/E$ =", err) 
                print(f"$\Delta \psi/\psi$ =", err) 
                # print(f"$\Delta \mu/\mu$ =", err) 
            # convergence checks
            if err < tolerance and use_tolerance:
                converged = True
                print("converged!")
                max_iter = i
                break
            if np.isnan(err):
                print("diverged, decrease timestep")
                break
            # noise addition
            if i < noise_iters and i % 20 == 0:
                rng = np.random.default_rng()
                x, y, z = grid
                Nx, Ny, Nz = np.size(x), np.size(y), np.size(z)
                Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
                dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
                if perlin:
                    res = (2**random.randint(0,6),2**random.randint(0,6),2**random.randint(0,6))
                    noise1 = np.array(generate_perlin_noise_3d(np.shape(x+y+z), res))
                    noise1 /= np.sqrt(np.sum(np.abs(noise1)**2)*dLx*dLy*dLz)
                    noise2 = np.array(generate_perlin_noise_3d(np.shape(x+y+z), res))
                    noise2 /= np.sqrt(np.sum(np.abs(noise2)**2)*dLx*dLy*dLz)
                    noise = np.stack((noise1, noise2),axis=0)
                    psi = psi + noise_level*noise
                    psi_norm = np.sqrt(np.sum(np.abs(psi)**2*dLx*dLy*dLz, axis=(1,2,3)))
                    psi = psi/psi_norm[:,np.newaxis,np.newaxis,np.newaxis]
                else:
                    noise = -1+2*(rng.random(np.shape(psi)))
                    psi = psi + psi*noise*noise_level
                    psi_norm = np.sqrt(np.sum(np.abs(psi)**2*dLx*dLy*dLz, axis=(1,2,3)))
                    psi = psi/psi_norm[:,np.newaxis,np.newaxis,np.newaxis]
            # if len(errs) > 1000 and err > np.max(errs[-10:-1]):
            #     print("failed to converge")
            #     max_iter = i
            #     break
    plt.close("all")
    # save results and plot energies 
    if save_name:
        filename = save_name+".npz"
        directory, _ = os.path.split(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if save_data:
            np.savez(filename, psi=psi, err=np.array(errs), mu1s=np.array(mu1s), mu2s=np.array(mu2s), Es=np.array(Es))
        # np.savez(filename, psi=psi)
        if energy_fn:
                plt.clf()
                plt.close("all")
                Es = np.array(Es)
                for i, label in enumerate(labels):
                    E_component = Es[:,i]
                    plt.plot(E_component.get(), label=label)
                plt.legend()
                file_path = directory+"/E.png"
                plt.savefig(file_path)
    return grid, psi, max_iter, errs, mu1s, mu2s, Es, converged


def run_simulation_rte(grid_shape, grid_boundaries, potential_propagator, kinetic_propagator, dt, iterations, guess1, guess2, iter_check=100, plot_all=False):
    grid,kgrid,psi = initialize_simulation(grid_shape, grid_boundaries, guess1, guess2)
    for i in range(iterations):
        psi, _ = real_time_propagation_step(psi,grid,kgrid,potential_propagator, kinetic_propagator,dt)
        if i % iter_check == 0:
            print("iteration:", i)
            if plot_all:
                plt_fns.plot_density_unnormalized_int(grid,np.abs(psi[0,:,:,:])**2, data_labels=("n1"))
                plt_fns.plot_density_unnormalized_int(grid,np.abs(psi[1,:,:,:])**2, data_labels=("n2"))

    return grid, psi


def set_up_animation(save_name,psi):
    writer1_context = writer2_context = nullcontext()
    filename = save_name+".npz"
    directory, _ = os.path.split(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig1, ax1 = plt.subplots()
    n1 = np.sum(np.abs(psi[0,:,:,:])**2,axis=2)
    im1 = ax1.imshow(n1.T)
    title1 = ax1.set_title("Iteration 0")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.colorbar(im1, ax=ax1, label="Integrated density")
    plt.tight_layout()
    writer1 = FFMpegWriter(
        fps=5,
        codec="libx264",
        bitrate=1800,
        metadata={"artist": "GPE solver"}
    )
    animation_name1 = directory+"/n1.mp4"
    writer1_context = writer1.saving(fig1, animation_name1, dpi=150) 
    fig2, ax2 = plt.subplots()
    n2 = np.sum(np.abs(psi[1,:,:,:])**2,axis=2)
    im2 = ax2.imshow(n2.T)
    title2 = ax2.set_title("Iteration 0")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    plt.colorbar(im2, ax=ax2, label="Integrated density")
    plt.tight_layout()
    writer2 = FFMpegWriter(
        fps=5,
        codec="libx264",
        bitrate=1800,
        metadata={"artist": "GPE solver"}
    )
    animation_name2 = directory+"/n2.mp4"
    writer2_context = writer2.saving(fig2, animation_name2, dpi=150)
    return (im1, title1, writer1, writer1_context), (im2, title2, writer2, writer2_context)


def initialize_simulation(grid_shape, grid_boundaries, guess1, guess2, plot_guess=False):
    xmin, xmax, ymin, ymax, zmin, zmax = np.ravel(grid_boundaries)
    x = np.linspace(xmin,xmax,num=grid_shape[0])
    y =  np.linspace(ymin,ymax,num=grid_shape[1])
    z =  np.linspace(zmin,zmax,num=grid_shape[2])
    Nx, Ny, Nz = grid_shape
    kx = 2*np.pi*np.fft.fftfreq(Nx)*Nx/(xmax-xmin)
    ky =  2*np.pi*np.fft.fftfreq(Ny)*Ny/(ymax-ymin)
    kz =  2*np.pi*np.fft.fftfreq(Nz)*Nz/(zmax-zmin)
    grid = np.meshgrid(x,y,z,sparse=True) 
    kgrid = np.meshgrid(kx,ky,kz,sparse=True) 
    psi = np.stack((guess1, guess2),axis=0)
    if plot_guess:
        plt_fns.plot_density_unnormalized_int(grid,np.abs(psi[0,:,:,:])**2)
        plt.show()
        plt_fns.plot_density_unnormalized_int(grid,np.abs(psi[1,:,:,:])**2)
        plt.show()
    print("simulation initialized")
    return (grid,kgrid,psi)


def imaginary_time_propagation_step(psi, grid, kgrid, potential_propagator, kinetic_propagator, dt):
    x, y, z = grid
    Nx, Ny, Nz = np.size(x), np.size(y), np.size(z)
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    # apply split step approach
    psi = kinetic_propagator(psi, grid, kgrid, dt/2)
    psi = potential_propagator(psi, grid, kgrid, dt)
    psi = kinetic_propagator(psi, grid, kgrid, dt/2)
    psi_norm = np.sqrt(np.sum(np.abs(psi)**2*dLx*dLy*dLz, axis=(1,2,3)))
    js = [0, 0]
    for i, norm in enumerate(psi_norm):
        j = js[i]
        # deal with posssible floating point overflow
        while ~ np.isfinite(norm) and j<10:
            psi[i,:,:,:] /= 1e50
            j += 1
    # renormalize psi
    psi = psi/psi_norm[:,np.newaxis,np.newaxis,np.newaxis]
    if js[0] != 0:
        psi_norm = np.array([np.inf, psi_norm[1]])
    if js[1] != 0:
        psi_norm = np.array([psi_norm[0], np.inf])
    return psi, psi_norm


def real_time_propagation_step(psi, grid, kgrid, potential_propagator, kinetic_propagator, dt):
    return imaginary_time_propagation_step(psi, grid, kgrid, potential_propagator, kinetic_propagator, 1j*dt)


### propagators
## propagators 
def single_dipolar_potential_propagator(psi, grid, kgrid, dt, gammas, anisotropy, pol_ratio, dipole_strength, dipole_direction, sag=0):
    # propagator for species 1 dipolar, species 2 nondipolar
    psi1 = psi[0,:,:,:]
    psi2 = psi[1,:,:,:]
    gamma11, gamma12, gamma21, gamma22 = gammas
    potential1 = dipolar_potential(psi1,grid,kgrid,gamma11,anisotropy,dipole_strength,dipole_direction)+gamma12*np.abs(psi2)**2
    potential2 = harmonic_potential(psi2,grid,kgrid,gamma22,anisotropy,scale_factor=pol_ratio,sag=sag)+gamma21*np.abs(psi1)**2
    return np.stack((np.exp(-dt*potential1)*psi1, np.exp(-dt*potential2)*psi2),axis=0)


def double_dipolar_potential_propagator(psi, grid, kgrid, dt, gammas, anisotropy, pol_ratio, dipole_strengths, dipole_direction, sag=0):
    # propagator for two dipolar species 
    psi1 = psi[0,:,:,:]
    psi2 = psi[1,:,:,:]
    gamma11, gamma12, gamma21, gamma22 = gammas
    d1, d12, d21, d2 = dipole_strengths
    kx, ky, kz = kgrid
    kabs = np.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    dipole_angle_fft = (3*costheta**2-1)
    dipole_angle_fft[kabs==0] = 0
    fft_n1 = fftw.fftn(np.abs(psi1)**2)
    fft_n2 = fftw.fftn(np.abs(psi2)**2)
    dipolar1 = fftw.ifftn((d1*fft_n1+d12*fft_n2)*dipole_angle_fft)
    dipolar2 = fftw.ifftn((d21*fft_n1+d2*fft_n2)*dipole_angle_fft)
    potential1 = dipolar1+harmonic_potential(psi1,grid,kgrid,gamma11,anisotropy)+gamma12*np.abs(psi2)**2
    potential2 = dipolar2+harmonic_potential(psi2,grid,kgrid,gamma22,anisotropy,pol_ratio,sag=sag)+gamma21*np.abs(psi1)**2
    return np.stack((np.exp(-dt*potential1)*psi1, np.exp(-dt*potential2)*psi2),axis=0)


def double_dipolar_potential_cutoff_propagator(psi, grid, kgrid, dt, gammas, anisotropy, pol_ratio, dipole_strengths, dipole_direction, sag=0):
    # propagator for two dipolar species, including a cutoff to remove spurious effects
    psi1 = psi[0,:,:,:]
    psi2 = psi[1,:,:,:]
    gamma11, gamma12, gamma21, gamma22 = gammas
    d1, d12, d21, d2 = dipole_strengths
    kx, ky, kz = kgrid
    x, y, z = grid
    cutoff_radius = max([float(np.max(x)), float(np.max(y)), float(np.max(z))])
    kabs = np.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    dipole_angle_fft = (1+3*np.cos(cutoff_radius*kabs)/(cutoff_radius*kabs)**2-3*np.sin(cutoff_radius*kabs)/(cutoff_radius*kabs)**3)*(3*costheta**2-1)
    dipole_angle_fft[kabs==0] = 0
    fft_n1 = fftw.fftn(np.abs(psi1)**2)
    fft_n2 = fftw.fftn(np.abs(psi2)**2)
    dipolar1 = fftw.ifftn((d1*fft_n1+d12*fft_n2)*dipole_angle_fft)
    dipolar2 = fftw.ifftn((d21*fft_n1+d2*fft_n2)*dipole_angle_fft)
    potential1 = dipolar1+harmonic_potential(psi1,grid,kgrid,gamma11,anisotropy)+gamma12*np.abs(psi2)**2
    potential2 = dipolar2+harmonic_potential(psi2,grid,kgrid,gamma22,anisotropy,pol_ratio,sag=sag)+gamma21*np.abs(psi1)**2
    return np.stack((np.exp(-dt*potential1)*psi1, np.exp(-dt*potential2)*psi2),axis=0)


def harmonic_potential_propagator(psi, grid, kgrid, dt, gammas, anisotropy, pol_ratio):
    # propagator for nondipolar species in harmonic trap
    psi1 = psi[0,:,:,:]
    psi2 = psi[1,:,:,:]
    gamma11, gamma12, gamma21, gamma22 = gammas
    potential1 = harmonic_potential(psi1,grid,kgrid,gamma11,anisotropy)+gamma12*np.abs(psi2)**2
    potential2 = harmonic_potential(psi2,grid,kgrid,gamma22,anisotropy,pol_ratio)+gamma21*np.abs(psi1)**2
    return np.stack((np.exp(-dt*potential1)*psi1, np.exp(-dt*potential2)*psi2),axis=0)


def box_potential_propagator(psi, grid, kgrid, dt, gammas, pad_fraction):
    # propagator for nondipolar species in box potential
    psi1 = psi[0,:,:,:]
    psi2 = psi[1,:,:,:]
    gamma11, gamma12, gamma21, gamma22 = gammas
    potential1 = gamma11*np.abs(psi1)**2+gamma12*np.abs(psi2)**2
    potential2 = gamma22*np.abs(psi2)**2+gamma21*np.abs(psi1)**2
    psi1_fill = np.exp(-dt*potential1)*psi1
    psi2_fill = np.exp(-dt*potential2)*psi2
    psi1 = np.zeros(np.shape(psi1_fill))
    psi2 = np.zeros(np.shape(psi2_fill))
    Nx, Ny, Nz = np.shape(psi1_fill)
    psi1[int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),int(Ny*pad_fraction/2):int(Ny*(1-pad_fraction/2)),int(Nz*pad_fraction/2):int(Nz*(1-pad_fraction/2)) ] = psi1_fill[int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),int(Ny*pad_fraction/2):int(Ny*(1-pad_fraction/2)),int(Nz*pad_fraction/2):int(Nz*(1-pad_fraction/2)) ]
    psi2[int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),int(Ny*pad_fraction/2):int(Ny*(1-pad_fraction/2)),int(Nz*pad_fraction/2):int(Nz*(1-pad_fraction/2)) ] = psi2_fill[int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),int(Ny*pad_fraction/2):int(Ny*(1-pad_fraction/2)),int(Nz*pad_fraction/2):int(Nz*(1-pad_fraction/2)) ]
    return np.stack((psi1, psi2),axis=0)


def dipole_potential_2D_box_propagator(psi, grid, kgrid, dt, gammas, pol_ratio, dipole_strengths, dipole_direction, pad_fraction):
    # propagator for dipolar species with harmonic trap in z and box in x, y; uses a cutoff
    psi1 = psi[0,:,:,:]
    psi2 = psi[1,:,:,:]
    d1, d12, d21, d2 = dipole_strengths
    kx, ky, kz = kgrid
    x, y, z = grid
    cutoff_radius = max([float(np.max(x)), float(np.max(y)), float(np.max(z))])
    kabs = np.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    dipole_angle_fft = (1+3*np.cos(cutoff_radius*kabs)/(cutoff_radius*kabs)**2-3*np.sin(cutoff_radius*kabs)/(cutoff_radius*kabs)**3)*(3*costheta**2-1)
    dipole_angle_fft[kabs==0] = 0
    fft_n1 = fftw.fftn(np.abs(psi1)**2)
    fft_n2 = fftw.fftn(np.abs(psi2)**2)
    dipolar1 = fftw.ifftn((d1*fft_n1+d12*fft_n2)*dipole_angle_fft)
    dipolar2 = fftw.ifftn((d21*fft_n1+d2*fft_n2)*dipole_angle_fft)
    gamma11, gamma12, gamma21, gamma22 = gammas
    potential1 = dipolar1+gamma11*np.abs(psi1)**2+gamma12*np.abs(psi2)**2+1/2*z**2
    potential2 = dipolar2+gamma22*np.abs(psi2)**2+gamma21*np.abs(psi1)**2+1/2*pol_ratio*z**2
    psi1_fill = np.exp(-dt*potential1)*psi1
    psi2_fill = np.exp(-dt*potential2)*psi2
    psi1 = np.zeros(np.shape(psi1_fill),dtype=complex)
    psi2 = np.zeros(np.shape(psi2_fill),dtype=complex)
    Nx, Ny, Nz = np.shape(psi1_fill)
    psi1[int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),int(Ny*pad_fraction/2):int(Ny*(1-pad_fraction/2)),:] = psi1_fill[int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),int(Ny*pad_fraction/2):int(Ny*(1-pad_fraction/2)),:]
    psi2[int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),int(Ny*pad_fraction/2):int(Ny*(1-pad_fraction/2)),:] = psi2_fill[int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),int(Ny*pad_fraction/2):int(Ny*(1-pad_fraction/2)),:]
    return np.stack((psi1, psi2),axis=0)


def box_kinetic_propagator(psi, grid, kgrid, dt, mass_ratio, pad_fraction):
    # kinetic energy propagator in a box
    fft_psi = np.fft.fftn(psi, axes=(1,2,3))
    kx, ky, kz = kgrid
    ksquared = kx**2+ky**2+kz**2
    ksquared = np.stack((ksquared, ksquared),axis=0)
    inverse_mass_factor = np.array([np.full(np.shape(psi)[1:],1),np.full(np.shape(psi)[1:], 1/mass_ratio)])
    psi =  np.fft.ifftn(np.exp(-1/2*inverse_mass_factor*ksquared*dt)*fft_psi, axes=(1,2,3))
    psi1_fill = psi[0,:,:,:]
    psi2_fill = psi[1,:,:,:]
    psi1 = np.zeros(np.shape(psi1_fill))
    psi2 = np.zeros(np.shape(psi2_fill))
    Nx, Ny, Nz = np.shape(psi1_fill)
    psi1[int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),int(Ny*pad_fraction/2):int(Ny*(1-pad_fraction/2)),int(Nz*pad_fraction/2):int(Nz*(1-pad_fraction/2)) ] = psi1_fill[int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),int(Ny*pad_fraction/2):int(Ny*(1-pad_fraction/2)),int(Nz*pad_fraction/2):int(Nz*(1-pad_fraction/2)) ]
    psi2[int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),int(Ny*pad_fraction/2):int(Ny*(1-pad_fraction/2)),int(Nz*pad_fraction/2):int(Nz*(1-pad_fraction/2)) ] = psi2_fill[int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),int(Ny*pad_fraction/2):int(Ny*(1-pad_fraction/2)),int(Nz*pad_fraction/2):int(Nz*(1-pad_fraction/2)) ]
    return np.stack((psi1, psi2),axis=0)


def box_2D_kinetic_propagator(psi, grid, kgrid, dt, mass_ratio, pad_fraction):
    # kinetic energy propagator with harmonic trap in z and box in x, y
    fft_psi = np.fft.fftn(psi, axes=(1,2,3))
    kx, ky, kz = kgrid
    ksquared = kx**2+ky**2+kz**2
    ksquared = np.stack((ksquared, ksquared),axis=0)
    inverse_mass_factor = np.array([np.full(np.shape(psi)[1:],1),np.full(np.shape(psi)[1:], 1/mass_ratio)])
    psi =  np.fft.ifftn(np.exp(-1/2*inverse_mass_factor*ksquared*dt)*fft_psi, axes=(1,2,3))
    psi1_fill = psi[0,:,:,:]
    psi2_fill = psi[1,:,:,:]
    psi1 = np.zeros(np.shape(psi1_fill),dtype=complex)
    psi2 = np.zeros(np.shape(psi2_fill),dtype=complex)
    Ny, Nx, Nz = np.shape(psi1_fill)
    psi1[int(Ny*pad_fraction/2):int(Ny*(1-pad_fraction/2)),int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),:] = psi1_fill[int(Ny*pad_fraction/2):int(Ny*(1-pad_fraction/2)),int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),:]
    psi2[int(Ny*pad_fraction/2):int(Ny*(1-pad_fraction/2)),int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),:] = psi2_fill[int(Ny*pad_fraction/2):int(Ny*(1-pad_fraction/2)),int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),:]
    return np.stack((psi1, psi2),axis=0)


def box_1D_kinetic_propagator(psi, grid, kgrid, dt, mass_ratio, pad_fraction):
     # kinetic energy propagator with harmonic trap in y, z and box in x
    fft_psi = np.fft.fftn(psi, axes=(1,2,3))
    kx, ky, kz = kgrid
    ksquared = kx**2+ky**2+kz**2
    ksquared = np.stack((ksquared, ksquared),axis=0)
    inverse_mass_factor = np.array([np.full(np.shape(psi)[1:],1),np.full(np.shape(psi)[1:], 1/mass_ratio)])
    psi =  np.fft.ifftn(np.exp(-1/2*inverse_mass_factor*ksquared*dt)*fft_psi, axes=(1,2,3))
    psi1_fill = psi[0,:,:,:]
    psi2_fill = psi[1,:,:,:]
    psi1 = np.zeros(np.shape(psi1_fill),dtype=complex)
    psi2 = np.zeros(np.shape(psi2_fill),dtype=complex)
    Ny, Nx, Nz = np.shape(psi1_fill)
    psi1[:,int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),:] = psi1_fill[:,int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),:]
    psi2[:,int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),:] = psi2_fill[:,int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),:]
    return np.stack((psi1, psi2),axis=0)


def dipole_potential_1D_box_propagator(psi, grid, kgrid, dt, gammas, pol_ratio, dipole_strengths, dipole_direction, pad_fraction):
    # propagator for dipolar species with harmonic trap in y, z and box in x; uses a cutoff
    psi1 = psi[0,:,:,:]
    psi2 = psi[1,:,:,:]
    d1, d12, d21, d2 = dipole_strengths
    kx, ky, kz = kgrid
    x, y, z = grid
    cutoff_radius = max([float(np.max(x)), float(np.max(y)), float(np.max(z))])
    kabs = np.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    dipole_angle_fft = (1+3*np.cos(cutoff_radius*kabs)/(cutoff_radius*kabs)**2-3*np.sin(cutoff_radius*kabs)/(cutoff_radius*kabs)**3)*(3*costheta**2-1)
    dipole_angle_fft[kabs==0] = 0
    fft_n1 = fftw.fftn(np.abs(psi1)**2)
    fft_n2 = fftw.fftn(np.abs(psi2)**2)
    dipolar1 = fftw.ifftn((d1*fft_n1+d12*fft_n2)*dipole_angle_fft)
    dipolar2 = fftw.ifftn((d21*fft_n1+d2*fft_n2)*dipole_angle_fft)
    gamma11, gamma12, gamma21, gamma22 = gammas
    potential1 = dipolar1+gamma11*np.abs(psi1)**2+gamma12*np.abs(psi2)**2+1/2*z**2+1/2*y**2
    potential2 = dipolar2+gamma22*np.abs(psi2)**2+gamma21*np.abs(psi1)**2+1/2*pol_ratio*z**2+1/2*pol_ratio*y**2
    psi1_fill = np.exp(-dt*potential1)*psi1
    psi2_fill = np.exp(-dt*potential2)*psi2
    psi1 = np.zeros(np.shape(psi1_fill),dtype=complex)
    psi2 = np.zeros(np.shape(psi2_fill),dtype=complex)
    Ny, Nx, Nz = np.shape(psi1_fill)
    psi1[:,int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),:] = psi1_fill[:,int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),:]
    psi2[:,int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),:] = psi2_fill[:,int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),:]
    return np.stack((psi1, psi2),axis=0)


def double_dipolar_potential_cutoff_intra_LHY_propagator(psi, grid, kgrid, dt, gammas, anisotropy, pol_ratio, dipole_strengths, dipole_direction, LHY_factors, sag=0):
    # propagator for two dipolar species + LHY correction calculated without interspecies term
    psi1 = psi[0,:,:,:]
    psi2 = psi[1,:,:,:]
    gamma11, gamma12, gamma21, gamma22 = gammas
    d1, d12, d21, d2 = dipole_strengths
    LHY1, LHY2 = LHY_factors
    kx, ky, kz = kgrid
    x, y, z = grid
    cutoff_radius = max([float(np.max(x)), float(np.max(y)), float(np.max(z))])
    kabs = np.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    dipole_angle_fft = (1+3*np.cos(cutoff_radius*kabs)/(cutoff_radius*kabs)**2-3*np.sin(cutoff_radius*kabs)/(cutoff_radius*kabs)**3)*(3*costheta**2-1)
    dipole_angle_fft[kabs==0] = 0
    fft_n1 = fftw.fftn(np.abs(psi1)**2)
    fft_n2 = fftw.fftn(np.abs(psi2)**2)
    dipolar1 = fftw.ifftn((d1*fft_n1+d12*fft_n2)*dipole_angle_fft)
    dipolar2 = fftw.ifftn((d21*fft_n1+d2*fft_n2)*dipole_angle_fft)
    potential1 = dipolar1+harmonic_potential(psi1,grid,kgrid,gamma11,anisotropy)+gamma12*np.abs(psi2)**2+LHY1*psi1**3
    potential2 = dipolar2+harmonic_potential(psi2,grid,kgrid,gamma22,anisotropy,pol_ratio,sag=sag)+gamma21*np.abs(psi1)**2+LHY2*psi2**3
    return np.stack((np.exp(-dt*potential1)*psi1, np.exp(-dt*potential2)*psi2),axis=0)


def dipole_potential_1D_box_harmonic_propagator(psi, grid, kgrid, dt, gammas, pol_ratio, f_ratio, dipole_strengths, dipole_direction, pad_fraction):
    # propagator for dipolar species with harmonic trap in y, z and box + harmonic trap in x; uses a cutoff
    psi1 = psi[0,:,:,:]
    psi2 = psi[1,:,:,:]
    d1, d12, d21, d2 = dipole_strengths
    kx, ky, kz = kgrid
    x, y, z = grid
    cutoff_radius = max([float(np.max(x)), float(np.max(y)), float(np.max(z))])
    kabs = np.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    dipole_angle_fft = (1+3*np.cos(cutoff_radius*kabs)/(cutoff_radius*kabs)**2-3*np.sin(cutoff_radius*kabs)/(cutoff_radius*kabs)**3)*(3*costheta**2-1)
    dipole_angle_fft[kabs==0] = 0
    fft_n1 = fftw.fftn(np.abs(psi1)**2)
    fft_n2 = fftw.fftn(np.abs(psi2)**2)
    dipolar1 = fftw.ifftn((d1*fft_n1+d12*fft_n2)*dipole_angle_fft)
    dipolar2 = fftw.ifftn((d21*fft_n1+d2*fft_n2)*dipole_angle_fft)
    gamma11, gamma12, gamma21, gamma22 = gammas
    potential1 = dipolar1+gamma11*np.abs(psi1)**2+gamma12*np.abs(psi2)**2+1/2*z**2+1/2*y**2+1/2*f_ratio**2*x**2
    potential2 = dipolar2+gamma22*np.abs(psi2)**2+gamma21*np.abs(psi1)**2+1/2*pol_ratio*z**2+1/2*pol_ratio*y**2+1/2*pol_ratio*f_ratio**2*x**2
    psi1_fill = np.exp(-dt*potential1)*psi1
    psi2_fill = np.exp(-dt*potential2)*psi2
    psi1 = np.zeros(np.shape(psi1_fill),dtype=complex)
    psi2 = np.zeros(np.shape(psi2_fill),dtype=complex)
    Ny, Nx, Nz = np.shape(psi1_fill)
    psi1[:,int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),:] = psi1_fill[:,int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),:]
    psi2[:,int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),:] = psi2_fill[:,int(Nx*pad_fraction/2):int(Nx*(1-pad_fraction/2)),:]
    return np.stack((psi1, psi2),axis=0)


def harmonic_potential(psi,grid,kgrid,gamma,anisotropy,scale_factor=1,sag=0):
    x, y, z = grid
    return 1/2*scale_factor*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*(z+sag)**2)+gamma*np.abs(psi)**2


def dipolar_potential(psi,grid,kgrid,gamma,anisotropy,dipole_strength,dipole_direction):
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
    return dipolar+harmonic_plus_contact


def mixture_kinetic_propagator(psi, grid, kgrid, dt, mass_ratio):
    # kinetic propagator for mixture in harmonic trap 
    fft_psi = np.fft.fftn(psi, axes=(1,2,3))
    kx, ky, kz = kgrid
    ksquared = kx**2+ky**2+kz**2
    ksquared = np.stack((ksquared, ksquared),axis=0)
    inverse_mass_factor = np.array([np.full(np.shape(psi)[1:],1),np.full(np.shape(psi)[1:], 1/mass_ratio)])
    return np.fft.ifftn(np.exp(-1/2*inverse_mass_factor*ksquared*dt)*fft_psi, axes=(1,2,3))

## energy functionals
def energy_functional_single_dipolar(psi,grid,kgrid,gammas,anisotropy, pol_ratio, mass_ratio, dipole_strength, dipole_direction, atom_nums, sag=0):
    N1, N2 = atom_nums
    fft_psi = np.fft.fftn(psi, axes=(1,2,3))
    x, y, z = grid
    Nx, Ny, Nz = np.size(x), np.size(y), np.size(z)
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    kx, ky, kz = kgrid
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    psi1 = psi[0,:,:,:]
    psi2 = psi[1,:,:,:]
    gamma11, gamma12, gamma21, gamma22 = gammas
    # kinetic energy 
    ksquared = kx**2+ky**2+kz**2
    ksquared = np.stack((ksquared, ksquared),axis=0)
    inverse_mass_factor = np.array([np.full(np.shape(psi)[1:],1),np.full(np.shape(psi)[1:], 1/mass_ratio)])
    KE_psi = np.fft.ifftn(1/2*inverse_mass_factor*ksquared*fft_psi, axes=(1,2,3))
    KE_psi1 = KE_psi[0,:,:,:]
    KE_psi2 = KE_psi[1,:,:,:]
    KE = np.sum((N1*np.conj(psi1)*KE_psi1+N2*np.conj(psi2)*KE_psi2)*dLx*dLy*dLz)
    # harmonic
    pot1 = 1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)
    pot2 = pol_ratio*1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*(z+sag)**2)
    harmonic =  np.sum((pot1*np.abs(psi1)**2*N1+pot2*np.abs(psi2)**2*N2)*dLx*dLy*dLz)
    # contact
    contact = np.sum((1/2*gamma11*N1*np.abs(psi1)**4+gamma12*N1*np.abs(psi1)**2*np.abs(psi2)**2+1/2*N2*gamma22*np.abs(psi2)**4)*dLx*dLy*dLz)
    # dipolar contribution
    fft_n = fftw.fftn(np.abs(psi1)**2)
    kx, ky, kz = kgrid
    kabs = np.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    dipole_fft = dipole_strength*(3*costheta**2-1)
    dipole_fft[kabs==0] = 0
    dipolar =  np.sum(1/2*fftw.ifftn(dipole_fft*fft_n)*N1*np.abs(psi1)**2*dLx*dLy*dLz)
    # turn into per-particle energy
    KE = np.real(KE/(N1+N2))
    harmonic = harmonic/(N1+N2)
    contact = contact/(N1+N2)
    dipolar = np.real(dipolar/(N1+N2))
    E = KE+harmonic+contact+dipolar
    labels = ("KE", "harmonic", "contact", "dipolar", "E")
    energies = (float(KE), float(harmonic), float(contact), float(dipolar), float(E))
    return labels, energies


def energy_functional_double_dipolar(psi,grid,kgrid,gammas,anisotropy, pol_ratio, mass_ratio, dipole_strengths, dipole_direction, atom_nums, sag=0):
    N1, N2 = atom_nums
    fft_psi = np.fft.fftn(psi, axes=(1,2,3))
    x, y, z = grid
    Nx, Ny, Nz = np.size(x), np.size(y), np.size(z)
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    kx, ky, kz = kgrid
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    psi1 = psi[0,:,:,:]
    psi2 = psi[1,:,:,:]
    gamma11, gamma12, gamma21, gamma22 = gammas
    d1, d12, d21, d2 = dipole_strengths
    # kinetic energy 
    ksquared = kx**2+ky**2+kz**2
    ksquared = np.stack((ksquared, ksquared),axis=0)
    inverse_mass_factor = np.array([np.full(np.shape(psi)[1:],1),np.full(np.shape(psi)[1:], 1/mass_ratio)])
    KE_psi = np.fft.ifftn(1/2*inverse_mass_factor*ksquared*fft_psi, axes=(1,2,3))
    KE_psi1 = KE_psi[0,:,:,:]
    KE_psi2 = KE_psi[1,:,:,:]
    KE = np.sum((N1*np.conj(psi1)*KE_psi1+N2*np.conj(psi2)*KE_psi2)*dLx*dLy*dLz)
    # harmonic
    pot1 = 1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)
    pot2 = pol_ratio*1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*(z+sag)**2)
    harmonic =  np.sum((pot1*np.abs(psi1)**2*N1+pot2*np.abs(psi2)**2*N2)*dLx*dLy*dLz)
    # contact
    contact = np.sum((1/2*gamma11*N1*np.abs(psi1)**4+gamma12*N1*np.abs(psi1)**2*np.abs(psi2)**2+1/2*N2*gamma22*np.abs(psi2)**4)*dLx*dLy*dLz)
    # dipolar contribution
    kabs = np.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    dipole_angle_fft = (3*costheta**2-1)
    dipole_angle_fft[kabs==0] = 0
    fft_n1 = fftw.fftn(np.abs(psi1)**2)
    fft_n2 = fftw.fftn(np.abs(psi2)**2)
    dipolar1_pot = fftw.ifftn((d1*fft_n1+d12*fft_n2)*dipole_angle_fft)*N1
    dipolar2_pot = fftw.ifftn((d21*fft_n1+d2*fft_n2)*dipole_angle_fft)*N2
    dipolar1 = np.sum(1/2*dipolar1_pot*np.abs(psi1)**2*dLx*dLy*dLz)/(N1+N2)
    dipolar2 = np.sum(1/2*dipolar2_pot*np.abs(psi2)**2*dLx*dLy*dLz)/(N1+N2)
    dipolar = dipolar1+dipolar2
    # turn into per-particle energy
    KE = np.real(KE/(N1+N2))
    harmonic = harmonic/(N1+N2)
    contact = contact/(N1+N2)
    dipolar = np.real(dipolar)
    E = KE+harmonic+contact+dipolar
    labels = ("KE", "harmonic", "contact", "dipolar", "E")
    energies = (float(KE), float(harmonic), float(contact), float(dipolar), float(E))
    return labels, energies


def energy_functional_double_dipolar_cutoff(psi,grid,kgrid,gammas,anisotropy, pol_ratio, mass_ratio, dipole_strengths, dipole_direction, atom_nums, sag=0):
    N1, N2 = atom_nums
    fft_psi = np.fft.fftn(psi, axes=(1,2,3))
    x, y, z = grid
    Nx, Ny, Nz = np.size(x), np.size(y), np.size(z)
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    kx, ky, kz = kgrid
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    psi1 = psi[0,:,:,:]
    psi2 = psi[1,:,:,:]
    gamma11, gamma12, gamma21, gamma22 = gammas
    d1, d12, d21, d2 = dipole_strengths
    # kinetic energy
    ksquared = kx**2+ky**2+kz**2
    ksquared = np.stack((ksquared, ksquared),axis=0)
    inverse_mass_factor = np.array([np.full(np.shape(psi)[1:],1),np.full(np.shape(psi)[1:], 1/mass_ratio)])
    KE_psi = np.fft.ifftn(1/2*inverse_mass_factor*ksquared*fft_psi, axes=(1,2,3))
    KE_psi1 = KE_psi[0,:,:,:]
    KE_psi2 = KE_psi[1,:,:,:]
    KE = np.sum((N1*np.conj(psi1)*KE_psi1+N2*np.conj(psi2)*KE_psi2)*dLx*dLy*dLz)
    # harmonic
    pot1 = 1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)
    pot2 = pol_ratio*1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*(z+sag)**2)
    harmonic =  np.sum((pot1*np.abs(psi1)**2*N1+pot2*np.abs(psi2)**2*N2)*dLx*dLy*dLz)
    # contact
    contact = np.sum((1/2*gamma11*N1*np.abs(psi1)**4+gamma12*N1*np.abs(psi1)**2*np.abs(psi2)**2+1/2*N2*gamma22*np.abs(psi2)**4)*dLx*dLy*dLz)
    # dipolar contribution
    cutoff_radius = max([float(np.max(x)), float(np.max(y)), float(np.max(z))])
    kabs = np.sqrt(kx**2+ky**2+kz**2)
    costheta = (kx*dipole_direction[0]+ky*dipole_direction[1]+kz*dipole_direction[2])/kabs
    dipole_angle_fft = (1+3*np.cos(cutoff_radius*kabs)/(cutoff_radius*kabs)**2-3*np.sin(cutoff_radius*kabs)/(cutoff_radius*kabs)**3)*(3*costheta**2-1)
    dipole_angle_fft[kabs==0] = 0
    fft_n1 = fftw.fftn(np.abs(psi1)**2)
    fft_n2 = fftw.fftn(np.abs(psi2)**2)
    dipolar1_pot = fftw.ifftn((d1*fft_n1+d12*fft_n2)*dipole_angle_fft)*N1
    dipolar2_pot = fftw.ifftn((d21*fft_n1+d2*fft_n2)*dipole_angle_fft)*N2
    dipolar1 = np.sum(1/2*dipolar1_pot*np.abs(psi1)**2*dLx*dLy*dLz)/(N1+N2)
    dipolar2 = np.sum(1/2*dipolar2_pot*np.abs(psi2)**2*dLx*dLy*dLz)/(N1+N2)
    # print("dipolar1: ", dipolar1)
    # print("dipolar2: ", dipolar2)
    dipolar = dipolar1+dipolar2
    # turn into per-particle energy
    KE = np.real(KE/(N1+N2))
    harmonic = harmonic/(N1+N2)
    contact = contact/(N1+N2)
    dipolar = np.real(dipolar)
    E = KE+harmonic+contact+dipolar
    # print("KE: ", KE)
    # print("harmonic: ", harmonic)
    # print("contact: ", contact)
    # print("dipolar: ", dipolar)
    # print("E: ", E)
    labels = ("KE", "harmonic", "contact", "dipolar", "E")
    energies = (float(KE), float(harmonic), float(contact), float(dipolar), float(E))
    return labels, energies


def energy_functional_double_dipolar_cutoff_intra_LHY(psi,grid,kgrid,gammas,anisotropy, pol_ratio, mass_ratio, dipole_strengths, dipole_direction, atom_nums, LHY_factors, sag=0):
    psi1 = psi[0,:,:,:]
    psi2 = psi[1,:,:,:]
    labels, energies = energy_functional_double_dipolar_cutoff(psi,grid,kgrid,gammas,anisotropy, pol_ratio, mass_ratio, dipole_strengths, dipole_direction, atom_nums, sag)
    x,y,z = grid
    N1, N2 = atom_nums
    LHY1, LHY2 = LHY_factors
    Nx, Ny, Nz = np.size(x), np.size(y), np.size(z)
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    E_LHY1 =np.sum(2/5*LHY1*np.abs(psi1)**5*dLx*dLy*dLz)*N1/(N1+N2)
    E_LHY2 =np.sum(2/5*LHY2*np.abs(psi2)**5*dLx*dLy*dLz)*N2/(N1+N2)
    KE, harmonic, contact, dipolar, E = energies
    E += float(E_LHY1+E_LHY2)
    labels = (*labels[:-1], "LHY", "E")
    energies = (*energies[:-1], float(E_LHY1+E_LHY2), E)
    return labels, energies

    
def Gaussian_ansatz(grid_shape, grid_boundaries, sigmax, sigmay, sigmaz, noise_level=0):
    """
    Gaussian profile + some random noise
    """
    xmin, xmax, ymin, ymax, zmin, zmax = grid_boundaries
    x = np.linspace(xmin,xmax,num=grid_shape[0])
    y =  np.linspace(ymin,ymax,num=grid_shape[1])
    z =  np.linspace(zmin,zmax,num=grid_shape[2])
    grid = np.meshgrid(x,y,z,sparse=True)
    x, y, z = grid
    Nx, Ny, Nz = np.size(x), np.size(y), np.size(z)
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    rng = np.random.default_rng()
    psi = np.exp(-2*((x**2)/(sigmax**2)+y**2/sigmay**2 + z**2/sigmaz**2))
    noise = -1+2*(rng.random(np.shape(psi)))
    psi = psi + psi*noise*noise_level
    random_phase = 2*np.pi*1j*noise_level*(rng.random(np.shape(psi)))
    psi = psi*np.exp(random_phase)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    psi_norm = np.sum(np.abs(psi)**2*dLx*dLy*dLz)
    return psi/np.sqrt(psi_norm)


def modulated_Gaussian(grid_shape, grid_boundaries, sigmax, sigmay, sigmaz, kx, ky, kz, noise_level=0):
    """
    Gaussian modulated by k = (kx, ky, kz)
    """
    xmin, xmax, ymin, ymax, zmin, zmax = grid_boundaries
    x = np.linspace(xmin,xmax,num=grid_shape[0])
    y =  np.linspace(ymin,ymax,num=grid_shape[1])
    z =  np.linspace(zmin,zmax,num=grid_shape[2])
    grid = np.meshgrid(x,y,z,sparse=True)
    x, y, z = grid
    Nx, Ny, Nz = np.size(x), np.size(y), np.size(z)
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    rng = np.random.default_rng()
    psi = np.exp(-2*((x**2)/(sigmax**2)+y**2/sigmay**2 + z**2/sigmaz**2))
    noise = -1+2*(rng.random(np.shape(psi)))
    psi = psi + psi*noise*noise_level
    psi = psi*np.exp(1j*(kx*x+ky*y+kz*z))
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    psi_norm = np.sum(np.abs(psi)**2*dLx*dLy*dLz)
    return psi/np.sqrt(psi_norm)


def random_guess(grid_shape, grid_boundaries):
    """
    Psi with randomized density profile in k space
    """
    xmin, xmax, ymin, ymax, zmin, zmax = grid_boundaries
    x = np.linspace(xmin,xmax,num=grid_shape[0])
    y =  np.linspace(ymin,ymax,num=grid_shape[1])
    z =  np.linspace(zmin,zmax,num=grid_shape[2])
    grid = np.meshgrid(x,y,z,sparse=True)
    x, y, z = grid
    Nx, Ny, Nz = np.size(x), np.size(y), np.size(z)
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    rng = np.random.default_rng()
    psi = rng.random(np.shape(x+y+z))*np.exp(2j*np.pi*rng.random(np.shape(x+y+z)))
    # psi = gaussian_filter(psi, 1)
    psi = fftw.fftn(psi)
    psi_norm = np.sqrt(np.sum(np.abs(psi)**2*dLx*dLy*dLz))
    return psi/psi_norm


def random_guess_sym(grid_shape, grid_boundaries):
    """
    Psi with randomized density profile in k space with x -> -x, y -> -y, z -> -z symmetry imposed
    """
    xmin, xmax, ymin, ymax, zmin, zmax = grid_boundaries
    x = np.linspace(xmin,xmax,num=grid_shape[0])
    y =  np.linspace(ymin,ymax,num=grid_shape[1])
    z =  np.linspace(zmin,zmax,num=grid_shape[2])
    grid = np.meshgrid(x,y,z,sparse=True)
    x, y, z = grid
    Nx, Ny, Nz = np.size(x), np.size(y), np.size(z)
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    rng = np.random.default_rng()
    psi = rng.random(np.shape(x+y+z))*np.exp(2j*np.pi*rng.random(np.shape(x+y+z)))
    # psi = gaussian_filter(psi, 1)
    psi = fftw.fftn(psi)
    psi = psi+np.flip(psi,axis=0)+np.flip(psi,axis=1)+np.flip(psi,axis=2)+np.flip(psi,axis=(0,1))+np.flip(psi,axis=(0,2))+np.flip(psi,axis=(1,2))+np.flip(psi,axis=(0,1,2))
    psi_norm = np.sqrt(np.sum(np.abs(psi)**2*dLx*dLy*dLz))
    return psi/psi_norm


def perlin_guess(grid_shape, grid_boundaries, res=(2,2,2)):
    """
    Perlin noise profile 
    """
    xmin, xmax, ymin, ymax, zmin, zmax = grid_boundaries
    x = np.linspace(xmin,xmax,num=grid_shape[0])
    y =  np.linspace(ymin,ymax,num=grid_shape[1])
    z =  np.linspace(zmin,zmax,num=grid_shape[2])
    grid = np.meshgrid(x,y,z,sparse=True)
    x, y, z = grid
    Nx, Ny, Nz = np.size(x), np.size(y), np.size(z)
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    psi = np.array(generate_perlin_noise_3d(np.shape(x+y+z), res))
    # psi = gaussian_filter(psi, 1)
    psi_norm = np.sqrt(np.sum(np.abs(psi)**2*dLx*dLy*dLz))
    return psi/psi_norm


def thomas_fermi_density(grid_shape, grid_boundaries, gamma,anisotropy):
    """Calculates Thomas Fermi profile
    gamma: gN scaled to correct units
    anisotropy: three element array specifying w^2 values along each axis"""
    xmin, xmax, ymin, ymax, zmin, zmax = np.ravel(grid_boundaries)
    x = np.linspace(xmin,xmax,num=grid_shape[0])
    y =  np.linspace(ymin,ymax,num=grid_shape[1])
    z =  np.linspace(zmin,zmax,num=grid_shape[2])
    Nx, Ny, Nz = grid_shape
    Lx, Ly, Lz = np.max(x)-np.min(x), np.max(y)-np.min(y),  np.max(z)-np.min(z)
    dLx, dLy, dLz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    x,y,z = np.meshgrid(x,y,z,sparse=True)
    potential = 1/2*(anisotropy[0]*x**2+anisotropy[1]*y**2+anisotropy[2]*z**2)
    mu = (15*gamma/(8*2**(3/2)*np.pi)*np.sqrt(anisotropy[0]*anisotropy[1]*anisotropy[2]))**(2/5)
    tf = (mu-potential)/gamma
    tf[tf<0] = 0
    tf_norm = np.sum(tf*dLx*dLy*dLz)
    return tf/tf_norm
