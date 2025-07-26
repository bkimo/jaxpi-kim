from functools import partial
import time
import os

from absl import logging

from flax.training import checkpoints

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, pmap
from jax.tree_util import tree_map

import scipy.io
import ml_collections

import wandb

import cylinder_flow

from jaxpi.utils import restore_checkpoint

from data_utils import get_dataset, parabolic_inflow

import matplotlib.pyplot as plt
import matplotlib.tri as tri


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Load dataset
    (
        u_ref,
        v_ref,
        p_ref,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cylinder_coords,
        nu,   # Check if we need nu? It might be better to remove because we have alpha term
    ) = get_dataset()


    T = 1.0  # final time

    # Nondimensionalization
    if config.nondim == True:
        # Nondimensionalization parameters
        U_star = 1.0  # characteristic velocity
        L_star = 0.1  # characteristic length
        T_star = L_star / U_star  # characteristic time
        Re = U_star * L_star / nu

        # Nondimensionalize coordinates and inflow velocity
        T = T / T_star
        inflow_coords = inflow_coords / L_star
        outflow_coords = outflow_coords / L_star
        wall_coords = wall_coords / L_star
        cylinder_coords = cylinder_coords / L_star
        coords = coords / L_star

        # Nondimensionalize flow field
        u_ref = u_ref / U_star
        v_ref = v_ref / U_star
        p_ref = p_ref / U_star**2

    else:
        Re = nu

    # Inflow boundary conditions
    U_max = 1.5  # maximum velocity
    inflow_fn = lambda y: parabolic_inflow(y * L_star, U_max)

    # Temporal domain of each time window
    t0 = 0.0
    t1 = 1.0

    temporal_dom = jnp.array([t0, t1 * (1 + 0.05)])  # Must be same as the one used in training

    # Initialize model
    model = cylinder_flow.LerayAlpha2D(config, inflow_fn, temporal_dom, coords, Re, alpha=config.alpha)

    # Restore checkpoint
    ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Predict
    u_pred_fn = jit(vmap(vmap(model.u_net, (None, None, 0, 0)), (None, 0, None, None)))
    v_pred_fn = jit(vmap(vmap(model.v_net, (None, None, 0, 0)), (None, 0, None, None)))
    p_pred_fn = jit(vmap(vmap(model.p_net, (None, None, 0, 0)), (None, 0, None, None)))
    # For the original unsteady example, there was a w_net for vorticity. 
    # Note that it's not directly part of the Leray-alpha PDE.
    w_pred_fn = jit(vmap(vmap(model.w_net, (None, None, 0, 0)), (None, 0, None, None)))

    # Create time steps for evaluation and plotting
    t_coords = jnp.linspace(0, t1, 100)  # More time points for better resolution

    u_pred_list = []
    v_pred_list = []
    p_pred_list = []
    w_pred_list = []
    C_D_list = []
    C_L_list = []
    p_diff_list = []

    # Collect predictions from each time window
    for idx in range(config.training.num_time_windows):
        # Try multiple possible checkpoint paths
        possible_paths = [
            os.path.abspath(os.path.join(workdir, config.wandb.name, "ckpt", f"time_window_{idx + 1}")),
            os.path.abspath(os.path.join(workdir, "ckpt", f"time_window_{idx + 1}")),
            os.path.abspath(os.path.join(workdir, "default", "ckpt", f"time_window_{idx + 1}")),
            os.path.abspath(os.path.join(workdir, config.wandb.name, "ckpt")),  # Single checkpoint case
            os.path.abspath(os.path.join(workdir, "ckpt")),  # Single checkpoint case
            os.path.abspath(os.path.join(workdir, "default", "ckpt"))  # Single checkpoint case
        ]
        
        current_ckpt_path = None
        for path in possible_paths:
            if os.path.isdir(path):
                current_ckpt_path = path
                break
        
        # Check if any valid directory was found
        if current_ckpt_path is None:
            logging.warning(f"No checkpoint directory found for time window {idx + 1}. Tried paths: {possible_paths}")
            continue

        try:
            model.state = restore_checkpoint(model.state, current_ckpt_path)
            params = model.state.params

            # Predict for the current time window
            u_pred = u_pred_fn(params, t_coords, coords[:, 0], coords[:, 1])
            v_pred = v_pred_fn(params, t_coords, coords[:, 0], coords[:, 1])
            w_pred = w_pred_fn(params, t_coords, coords[:, 0], coords[:, 1])
            p_pred = p_pred_fn(params, t_coords, coords[:, 0], coords[:, 1])

            # Compute drag, lift, and pressure drop for each time step
            C_D, C_L, p_diff = model.compute_drag_lift(params, t_coords, U_star, L_star)

            u_pred_list.append(u_pred)
            v_pred_list.append(v_pred)
            w_pred_list.append(w_pred)
            p_pred_list.append(p_pred)
            C_D_list.append(C_D)
            C_L_list.append(C_L)
            p_diff_list.append(p_diff)
            
            logging.info(f"Successfully loaded checkpoint from: {current_ckpt_path}")
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint from {current_ckpt_path}: {e}")
            continue

    # Check if any predictions were collected before concatenation
    if len(u_pred_list) == 0:
        raise ValueError(f"No valid checkpoints found! Please check that checkpoints exist in the expected directories. "
                        f"Looked for checkpoints with num_time_windows={config.training.num_time_windows} in workdir='{workdir}'")

    # Concatenate results from all time windows
    u_pred = jnp.concatenate(u_pred_list, axis=0)
    v_pred = jnp.concatenate(v_pred_list, axis=0)
    p_pred = jnp.concatenate(p_pred_list, axis=0)
    w_pred = jnp.concatenate(w_pred_list, axis=0)
    C_D = jnp.concatenate(C_D_list, axis=0)
    C_L = jnp.concatenate(C_L_list, axis=0)
    p_diff = jnp.concatenate(p_diff_list, axis=0)

    # Compute L2 errors for the final time step
    u_error = jnp.sqrt(jnp.mean((u_ref[-1] - u_pred[-1]) ** 2)) / jnp.sqrt(jnp.mean(u_ref[-1]**2))
    v_error = jnp.sqrt(jnp.mean((v_ref[-1] - v_pred[-1]) ** 2)) / jnp.sqrt(jnp.mean(v_ref[-1]**2))

    print("l2_error of u: {:.4e}".format(u_error))
    print("l2_error of v: {:.4e}".format(v_error))

    # Compute maximum values for non-steady flow
    C_D_max = jnp.max(C_D)
    C_L_max = jnp.max(jnp.abs(C_L))  # Use absolute value for lift
    p_diff_max = jnp.max(jnp.abs(p_diff))  # Use absolute value for pressure difference

    # Reference values (these would need to be updated with appropriate reference values for Leray-alpha)
    # For now, using placeholder values - these should be replaced with actual reference values
    C_D_ref = 3.2  # Placeholder reference value
    C_L_ref = 0.5  # Placeholder reference value  
    p_diff_ref = 0.1  # Placeholder reference value

    C_D_error = jnp.abs(C_D_max - C_D_ref) / C_D_ref
    C_L_error = jnp.abs(C_L_max - C_L_ref) / C_L_ref
    p_diff_error = jnp.abs(p_diff_max - p_diff_ref) / p_diff_ref

    print("C_D_max: {:.4e}".format(C_D_max))
    print("C_L_max: {:.4e}".format(C_L_max))
    print("p_diff_max: {:.4e}".format(p_diff_max))
    print("Relative error of C_D_max: {:.4e}".format(C_D_error))
    print("Relative error of C_L_max: {:.4e}".format(C_L_error))
    print("Relative error of p_diff_max: {:.4e}".format(p_diff_error))

    # Dimensionalize coordinates and flow field 
    if config.nondim == True:
        # Dimensionalize coordinates and flow field
        coords = coords * L_star

        u_ref = u_ref * U_star
        v_ref = v_ref * U_star
        p_ref = p_ref * U_star**2

        u_pred = u_pred * U_star
        v_pred = v_pred * U_star
        p_pred = p_pred * U_star**2

    # Triangulation
    x = coords[:, 0]
    y = coords[:, 1]
    triang = tri.Triangulation(x, y)

    # Mask the triangles inside the cylinder
    center = (0.2, 0.2)
    radius = 0.05

    x_tri = x[triang.triangles].mean(axis=1)
    y_tri = y[triang.triangles].mean(axis=1)
    dist_from_center = jnp.sqrt((x_tri - center[0]) ** 2 + (y_tri - center[1]) ** 2)
    triang.set_mask(dist_from_center < radius)

    # Save dir for figures
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Plot and save the main prediction figure
    fig1 = plt.figure(figsize=(18, 12))
    plt.subplot(3, 1, 1)
    plt.tricontourf(triang, u_pred[-1], cmap='jet', levels=100)
    plt.colorbar()
    plt.title('Predicted $u$')
    plt.tight_layout()

    plt.subplot(3, 1, 2)
    plt.tricontourf(triang, v_pred[-1], cmap='jet', levels=100)
    plt.colorbar()
    plt.title('Predicted $v$')
    plt.tight_layout()

    plt.subplot(3, 1, 3)
    plt.tricontourf(triang, p_pred[-1], cmap='jet', levels=100)
    plt.colorbar()
    plt.title('Predicted $p$')
    plt.tight_layout()

    save_path = os.path.join(save_dir, "leray_alpha_cylinder.pdf")
    fig1.savefig(save_path, bbox_inches="tight", dpi=300)

    # Plot and save the error figure (reference - prediction)
    fig2 = plt.figure(figsize=(18, 12))
    plt.subplot(3, 1, 1)
    plt.tricontourf(triang, u_ref[-1] - u_pred[-1], cmap='bwr', levels=100)
    plt.colorbar()
    plt.title('Error $u_{ref} - u_{pred}$')
    plt.tight_layout()

    plt.subplot(3, 1, 2)
    plt.tricontourf(triang, v_ref[-1] - v_pred[-1], cmap='bwr', levels=100)
    plt.colorbar()
    plt.title('Error $v_{ref} - v_{pred}$')
    plt.tight_layout()

    plt.subplot(3, 1, 3)
    plt.tricontourf(triang, p_ref[-1] - p_pred[-1], cmap='bwr', levels=100)
    plt.colorbar()
    plt.title('Error $p_{ref} - p_{pred}$')
    plt.tight_layout()

    error_save_path = os.path.join(save_dir, "leray_alpha_cylinder_errors.pdf")
    fig2.savefig(error_save_path, bbox_inches="tight", dpi=300)

    # Plot and save the reference figure (u_ref, v_ref, p_ref)
    fig3 = plt.figure(figsize=(18, 12))
    plt.subplot(3, 1, 1)
    plt.tricontourf(triang, u_ref[-1], cmap='jet', levels=100)
    plt.colorbar()
    plt.title('Reference $u_{ref}$')
    plt.tight_layout()

    plt.subplot(3, 1, 2)
    plt.tricontourf(triang, v_ref[-1], cmap='jet', levels=100)
    plt.colorbar()
    plt.title('Reference $v_{ref}$')
    plt.tight_layout()

    plt.subplot(3, 1, 3)
    plt.tricontourf(triang, p_ref[-1], cmap='jet', levels=100)
    plt.colorbar()
    plt.title('Reference $p_{ref}$')
    plt.tight_layout()

    ref_save_path = os.path.join(save_dir, "ns_cylinder_reference.pdf")
    fig3.savefig(ref_save_path, bbox_inches="tight", dpi=300)

    # Create time series for plotting
    t_plot = jnp.linspace(0, t1, len(C_D))

    # Plot 1: Complete time interval [t0, t1]
    fig4, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # C_D over time
    axes[0].plot(t_plot, C_D, 'r-', linewidth=2, label=f'C_D (max: {C_D_max:.4f})')
    axes[0].set_xlabel('time')
    axes[0].set_ylabel('C_D')
    axes[0].set_title('Drag Coefficient vs Time')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # C_L over time
    axes[1].plot(t_plot, C_L, 'b-', linewidth=2, label=f'C_L (max: {C_L_max:.4f})')
    axes[1].set_xlabel('time')
    axes[1].set_ylabel('C_L')
    axes[1].set_title('Lift Coefficient vs Time')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # p_diff over time
    axes[2].plot(t_plot, p_diff, 'g-', linewidth=2, label=f'p_diff (max: {p_diff_max:.4f})')
    axes[2].set_xlabel('time')
    axes[2].set_ylabel('p_diff')
    axes[2].set_title('Pressure Difference vs Time')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    time_series_save_path = os.path.join(save_dir, "leray_alpha_time_series.pdf")
    fig4.savefig(time_series_save_path, bbox_inches="tight", dpi=300)

    # Plot 2: Near maximum values (zoom in around the maximum)
    # Find the time indices near the maximum values
    C_D_max_idx = jnp.argmax(C_D)
    C_L_max_idx = jnp.argmax(jnp.abs(C_L))
    p_diff_max_idx = jnp.argmax(jnp.abs(p_diff))
    
    # Create zoomed plots around maxima
    window_size = 20  # Number of points around maximum to show
    
    fig5, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # C_D near maximum
    start_idx = max(0, C_D_max_idx - window_size//2)
    end_idx = min(len(C_D), C_D_max_idx + window_size//2)
    t_zoom_C_D = t_plot[start_idx:end_idx]
    C_D_zoom = C_D[start_idx:end_idx]
    
    axes[0].plot(t_zoom_C_D, C_D_zoom, 'r-', linewidth=2, marker='o', markersize=4)
    axes[0].axvline(t_plot[C_D_max_idx], color='red', linestyle='--', alpha=0.7, label=f'Max C_D = {C_D_max:.4f}')
    axes[0].set_xlabel('time')
    axes[0].set_ylabel('C_D')
    axes[0].set_title('C_D Near Maximum Value')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # C_L near maximum
    start_idx = max(0, C_L_max_idx - window_size//2)
    end_idx = min(len(C_L), C_L_max_idx + window_size//2)
    t_zoom_C_L = t_plot[start_idx:end_idx]
    C_L_zoom = C_L[start_idx:end_idx]
    
    axes[1].plot(t_zoom_C_L, C_L_zoom, 'b-', linewidth=2, marker='o', markersize=4)
    axes[1].axvline(t_plot[C_L_max_idx], color='blue', linestyle='--', alpha=0.7, label=f'Max |C_L| = {C_L_max:.4f}')
    axes[1].set_xlabel('time')
    axes[1].set_ylabel('C_L')
    axes[1].set_title('C_L Near Maximum Value')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # p_diff near maximum
    start_idx = max(0, p_diff_max_idx - window_size//2)
    end_idx = min(len(p_diff), p_diff_max_idx + window_size//2)
    t_zoom_p_diff = t_plot[start_idx:end_idx]
    p_diff_zoom = p_diff[start_idx:end_idx]
    
    axes[2].plot(t_zoom_p_diff, p_diff_zoom, 'g-', linewidth=2, marker='o', markersize=4)
    axes[2].axvline(t_plot[p_diff_max_idx], color='green', linestyle='--', alpha=0.7, label=f'Max |p_diff| = {p_diff_max:.4f}')
    axes[2].set_xlabel('time')
    axes[2].set_ylabel('p_diff')
    axes[2].set_title('Pressure Difference Near Maximum Value')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    zoom_save_path = os.path.join(save_dir, "leray_alpha_zoom_maxima.pdf")
    fig5.savefig(zoom_save_path, bbox_inches="tight", dpi=300)

    # Combined plot similar to the reference image
    fig6, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all parameters on the same graph with different colors and styles
    ax.plot(t_plot, C_D, 'r-', linewidth=2, label=f'C_D (max: {C_D_max:.4f})')
    ax.plot(t_plot, C_L, 'b-', linewidth=2, label=f'C_L (max: {C_L_max:.4f})')
    ax.plot(t_plot, p_diff, 'g-', linewidth=2, label=f'p_diff (max: {p_diff_max:.4f})')
    
    ax.set_xlabel('time')
    ax.set_ylabel('Coefficient Values')
    ax.set_title('Leray-Alpha Cylinder Flow: Drag, Lift, and Pressure Difference vs Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    combined_save_path = os.path.join(save_dir, "leray_alpha_combined_parameters.pdf")
    fig6.savefig(combined_save_path, bbox_inches="tight", dpi=300)

    print(f"All figures saved to: {save_dir}")
    print(f"Time series plots: {time_series_save_path}")
    print(f"Zoom plots: {zoom_save_path}")
    print(f"Combined plot: {combined_save_path}")
