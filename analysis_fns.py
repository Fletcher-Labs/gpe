import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import os

## plotting
def plot_density_unnormalized_int(grid, data, sum_axes=("y",), data_labels=("n"), save_name=None, save_folder=None, units=None, show=True, title=None):
    if not isinstance(data, list):
        data = [data]
    sum_mapping = {"x": 1, "y": 0, "z": 2}
    sum_mapping_inv = {1: "x", 0: "y", 2: "z"}
    # sum_ax_inds = tuple(sum_mapping[ax] for ax in sum_axes)
    grid_mapping = {"x": 0, "y": 1, "z": 2}
    data_int = [] 
    for d in data:
        idx = [slice(None)] * 3
        for ax_name in sum_axes:
            coord = np.ravel(grid[grid_mapping[ax_name]])
            idx[sum_mapping[ax_name]] = int(np.argmin(np.abs(coord)))
        data_int.append(d[tuple(idx)])
    plot_axes = [sum_mapping_inv[ax_ind] for ax_ind in range(3) if sum_mapping_inv[ax_ind] not in sum_axes]
    plot_ax_inds = [grid_mapping[ax] for ax in plot_axes]
    fig, ax = plt.subplots()
    unit_factor = 1 if not units else units[0]
    unit_name = "lx" if not units else units[1]
    if len(plot_ax_inds) == 1:
        coords = np.ravel(grid[plot_ax_inds[0]])
        for i, d in enumerate(data_int):
            ax.plot(coords*unit_factor, d, label=data_labels[i])
        ax.set_xlabel(plot_axes[0] + " (" + unit_name + ")")
        ax.set_ylabel("Density")
        if title:
            ax.set_title(title)
        ax.legend()
        ax.grid(which='both', alpha=0.25)
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
            im = ax.pcolormesh(coords_x*unit_factor, unit_factor*coords_y, d.T, label=data_labels[i], vmax=m, vmin=0)
            ax.set_xlabel(plot_axes[0]+ " (" + unit_name + ")")
            ax.set_ylabel(plot_axes[1]+ " (" + unit_name + ")")
            fig.colorbar(im, fraction=0.046, pad=0.04)
            if len(data_labels) == len(data_int):
                ax.set_title(data_labels[i])
            if title:
                ax.set_title(title)
            if save_name and save_folder:
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                full_path = os.path.join(save_folder, save_name) # add back str(i) + 
                fig.savefig(full_path)
            if show:
                plt.show()
    else:
        raise ValueError("can only plot up to 2D")
    return data_int

def plot_3D(grid, psi, potential, density_mode="integrated", density_height=None, potential_max=None, units=None, save_name=None, save_folder=None, show=True):
    to_cpu = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)  # cupy -> numpy
    x, y, z = [to_cpu(a).ravel() for a in grid]
    psi = to_cpu(psi)  # shape (Ny, Nx, Nz)

    # extract coordinate vectors from vectors or meshgrid arrays.
    axes = []
    for coordinate, axis in zip(grid, (1, 0, 2)):
        a = to_cpu(coordinate)
        if a.ndim == 1:
            axes.append(a)
        else:
            index = [0, 0, 0]
            index[axis] = slice(None)
            axes.append(np.broadcast_to(a, psi.shape)[tuple(index)])

    x, y, z = axes
    n3 = np.abs(psi) ** 2
    if density_mode == "integrated":
        n2 = np.sum(n3, axis=2)
    elif density_mode == "slice":
        n2 = n3[:, :, np.argmin(np.abs(z))]
    else:
        raise ValueError("density_mode must be 'integrated' or 'slice'.")

    if callable(potential):
        coordinates = np.meshgrid(x, y, z, indexing="xy", sparse=True)
        potential = potential(tuple(np.array(a) for a in coordinates))

    V = to_cpu(potential)
    if V.ndim <= 2:
        V = np.broadcast_to(V, psi.shape[:2])
    elif V.ndim == 3:
        V = np.broadcast_to(V, psi.shape)[:, :, np.argmin(np.abs(z))]
    else:
        raise ValueError("potential must be scalar, 2D, or 3D.")

    if potential_max is None:
        potential_max = float(V.max())
    lower = min(0.0, float(V.min()))
    span = potential_max - lower
    if span < 0:
        raise ValueError("potential_max is below the plotted range.")
    if density_height is None:
        density_height = (span if span > 0 else 1.0) / 2

    peak = n2.max()
    normalized = n2 / peak if peak > 0 else np.zeros_like(n2)
    n_plot = density_height * normalized

    f, uname = (1.0, "l") if units is None else (float(units[0]), units[1])
    xs, ys = x * f, y * f

    fig = go.Figure()
    fig.add_trace(go.Surface(x=xs, y=ys, z=V, colorscale=[[0, "crimson"], [1, "crimson"]], showscale=False, opacity=0.35, name=f"V(x, y, z={z[np.argmin(np.abs(z))]:.3g})", 
                            showlegend=True, hovertemplate=("x=%{x:.3g}<br>y=%{y:.3g}<br>" "V=%{z:.3g} ħωz<extra>potential</extra>"),))
    fig.add_trace(go.Surface(x=xs, y=ys, z=n_plot, colorscale="Blues", showscale=False, opacity=0.9, name="Density (rescaled)", showlegend=True, customdata=normalized, 
                             hovertemplate=("x=%{x:.3g}<br>y=%{y:.3g}<br>" "n/n_max=%{customdata:.3g}<extra>density</extra>"),))
    upper = max(potential_max, float(n_plot.max()), 0.0)
    if upper == lower:
        upper = lower + 1.0
    fig.update_layout(scene=dict(xaxis_title=f"x ({uname})", yaxis_title=f"y ({uname})", zaxis_title="V / Density (a.u.)", zaxis=dict(range=[lower, upper]), 
                                 aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.6),), legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=30, b=0), title="Potential (red) and Density (blue)",)

    if save_name and save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        full_path = os.path.join(save_folder, save_name)
        fig.write_html(full_path, include_plotlyjs="cdn")
    if show:
        fig.show()
    return fig


def plot_density_dual_species_int(grid, data, sum_axes=("y",), data_labels=("n1", "n2"), dLs=[1,1,1], save_name=None, save_folder=None, units=None, show=True):
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


# analysis
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
