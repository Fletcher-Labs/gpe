import cupy as cp
from matplotlib import pyplot as plt
import os

def plot_density_unnormalized_int (grid, data, sum_axes=("z",), data_labels=("n"), title = "n", save_name=None, save_folder=None, units=None):
    unit_factor = 1 if not units else float(units[0])
    unit_name = "lx" if not units else units[1]

    if not isinstance(data,list):
        data = [data]
    sum_mapping = {"x": 1, "y": 0, "z": 2}
    sum_mapping_inv = {1: "x", 0: "y", 2: "z"}
    sum_ax_inds = tuple(sum_mapping[ax] for ax in sum_axes)
    data_int = []
    for d in data:
        data_int.append(cp.sum(d, axis=sum_ax_inds))
    grid_mapping = {"x":0, "y":1, "z":2}
    plot_axes = [sum_mapping_inv[ax_ind] for ax_ind in range(3) if sum_mapping_inv[ax_ind] not in sum_axes]
    plot_ax_inds = [grid_mapping[ax] for ax in plot_axes]
    fig, ax = plt.subplots()
    if len(plot_ax_inds) == 1:
        coords = cp.ravel(grid[plot_ax_inds[0]].get())
        for i,d in enumerate(data_int):
            ax.plot(coords*unit_factor,d,label=data_labels[i])
        ax.set_xlabel(plot_axes[0]+ "(" + unit_name + ")")
        ax.set_title(title)
        ax.legend()
        if save_name and save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            full_path = os.path.join(save_folder, save_name)
            fig.savefig(full_path)
    elif len(plot_ax_inds) == 2:
        coords_x = cp.ravel(grid[plot_ax_inds[0]].get())
        coords_y = cp.ravel(grid[plot_ax_inds[1]].get())
        for i,d in enumerate(data_int):
            ax.pcolormesh(coords_x*unit_factor, coords_y*unit_factor,d.T,label=data_labels[i])
        ax.set_xlabel(plot_axes[0]+ "(" + unit_name + ")")
        ax.set_ylabel(plot_axes[1]+ "(" + unit_name + ")")
        ax.set_title(title)
        ax.legend()
        if save_name and save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            full_path = os.path.join(save_folder, save_name)
            fig.savefig(full_path)
    else:
        raise ValueError("can only plot up to 2D")
    return fig