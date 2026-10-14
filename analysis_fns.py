import numpy as np
from matplotlib import pyplot as plt
import os

## plotting
def plot_density_unnormalized_int (grid, data, sum_axes=("y",), data_labels=("n"), save_name=None, save_folder=None, units=None, show=True):
    if not isinstance(data,list):
        data = [data]
    sum_mapping = {"x": 1, "y": 0, "z": 2}
    sum_mapping_inv = {1: "x", 0: "y", 2: "z"}
    sum_ax_inds = tuple(sum_mapping[ax] for ax in sum_axes)
    data_int = []
    for d in data:
        data_int.append(np.sum(d, axis=sum_ax_inds))
    grid_mapping = {"x":0, "y":1, "z":2}
    plot_axes = [sum_mapping_inv[ax_ind] for ax_ind in range(3) if sum_mapping_inv[ax_ind] not in sum_axes]
    plot_ax_inds = [grid_mapping[ax] for ax in plot_axes]
    fig, ax = plt.subplots()
    unit_factor = 1 if not units else units[0]
    unit_name = "lx" if not units else units[1]
    if len(plot_ax_inds) == 1:
        coords = np.ravel(grid[plot_ax_inds[0]])
        for i,d in enumerate(data_int):
            ax.plot(coords*unit_factor,d,label=data_labels[i])
        ax.set_xlabel(plot_axes[0] + "(" + unit_name + ")")
        ax.legend()
        if save_name and save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            full_path = os.path.join(save_folder, save_name)
            fig.savefig(full_path)
        if show:
            plt.show()
    elif len(plot_ax_inds) == 2:
        coords_x = np.ravel(grid[plot_ax_inds[0]])
        coords_y = np.ravel(grid[plot_ax_inds[1]])
        m = np.max(data_int)
        for i,d in enumerate(data_int):
            plt.close()
            fig, ax = plt.subplots()
            ax.pcolormesh(coords_x*unit_factor, unit_factor*coords_y,d.T,label=data_labels[i],vmax=m,vmin=0)
            ax.set_xlabel(plot_axes[0]+ "(" + unit_name + ")")
            ax.set_ylabel(plot_axes[1]+ "(" + unit_name + ")")
            if len(data_labels) == len(data_int):
                ax.set_title(data_labels[i])
            if save_name and save_folder:
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                full_path = os.path.join(save_folder, str(i)+save_name)
                fig.savefig(full_path)
            if show:
                plt.show()
    else:
        raise ValueError("can only plot up to 2D")
    return data_int


def plot_density_dual_species_int (grid, data, sum_axes=("y",), data_labels=("n1", "n2"), dLs= [1,1,1], save_name=None, save_folder=None, units=None, show=True):
    # function for plotting two species on same plot 
    if not isinstance(data,list) or len(data) != 2:
        raise ValueError("must provide two sets of data")
    sum_mapping = {"x": 1, "y": 0, "z": 2}
    sum_mapping_inv = {1: "x", 0: "y", 2: "z"}
    sum_ax_inds = tuple(sum_mapping[ax] for ax in sum_axes)
    data_int = []
    grid_mapping = {"x":0, "y":1, "z":2}
    dL_sum = [dLs[grid_mapping[ax]] for ax in sum_axes]
    dL = 1
    for dLx in dL_sum:
        dL *= dLx
    for d in data:
        data_int.append(np.sum(d*dL, axis=sum_ax_inds))
    plot_axes = [sum_mapping_inv[ax_ind] for ax_ind in range(3) if sum_mapping_inv[ax_ind] not in sum_axes]
    plot_ax_inds = [grid_mapping[ax] for ax in plot_axes]
    fig, ax = plt.subplots()
    unit_factor = 1 if not units else float(units[0])
    unit_name = "lx" if not units else units[1]
    if len(plot_ax_inds) == 1:
        coords = np.ravel(grid[plot_ax_inds[0]])
        colors = ["blue", "red"]
        for i,d in enumerate(data_int):
            ax.plot(coords*unit_factor,d,label=data_labels[i], color=colors[i])
        ax.set_xlabel(plot_axes[0] + "(" + unit_name + ")")
        ax.legend()
        if save_name and save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            full_path = os.path.join(save_folder, save_name)
            fig.savefig(full_path)
        if show:
            plt.show()
    elif len(plot_ax_inds) == 2:
        coords_x = np.ravel(grid[plot_ax_inds[0]])
        coords_y = np.ravel(grid[plot_ax_inds[1]])
        color_maps = ["Blues", "Reds"]
        locs = ["left", "right"]
        alphas = [1,0.6]
        fig, ax = plt.subplots()
        for i,d in enumerate(data_int):
            f = ax.pcolormesh(coords_x*unit_factor, unit_factor*coords_y,d.T,label=data_labels[i],vmin=0, cmap=color_maps[i], alpha=alphas[i])
            # fig.colorbar(f, location=locs[i])
            ax.set_xlabel(plot_axes[0]+ "(" + unit_name + ")")
            ax.set_ylabel(plot_axes[1]+ "(" + unit_name + ")")
        if save_name and save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            full_path = os.path.join(save_folder, save_name)
            fig.savefig(full_path)
        if show:
            plt.show()
    else:
        raise ValueError("can only plot up to 2D")
    return data_int


## analysis
def contrast(n1, axes=("y","z")):
    axis_mapping =  {"x": 1, "y": 0, "z": 2}
    axes = tuple(axis_mapping[axis] for axis in axes)
    nmin = np.max(n1, axis=axes)
    nmax = np.max(n1)
    Cs = (nmax-nmin)/(nmax+nmin)
    droplet_indices = np.where(Cs<0.2)[0]
    left_droplet = int(np.min(droplet_indices))
    right_droplet = int(np.max(droplet_indices))
    C = np.max(Cs[left_droplet:right_droplet])
    return Cs, C
    
def leggett_superfluid_fraction(n1, n2, axes=("y","z")):
    # calculates Leggett bound on superfluid fraction
    axis_mapping =  {"x": 1, "y": 0, "z": 2}
    axes = tuple(axis_mapping[axis] for axis in axes)
    n1_int = np.sum(n1, axis=axes)
    n2_int = np.sum(n2, axis=axes)
    # define central density region 
    center_indices1 = np.where(np.abs(n1_int)>0.2*np.max(np.abs(n1_int)))[0]
    left_center1 = int(np.min(center_indices1))
    right_center1 = int(np.max(center_indices1))
    n1_int = n1_int[left_center1:right_center1]
    # define central density region 
    center_indices2 = np.where(np.abs(n2_int)>0.2*np.max(np.abs(n2_int)))[0]
    left_center2 = int(np.min(center_indices2))
    right_center2 = int(np.max(center_indices2))
    n2_int = n2_int[left_center2:right_center2]
    # calculate superfluid fractions
    f1 = np.size(n1_int)**2/(np.sum(1/n1_int)*np.sum(n1_int))
    f2 = np.size(n2_int)**2/(np.sum(1/n2_int)*np.sum(n2_int))
    return f1, f2
