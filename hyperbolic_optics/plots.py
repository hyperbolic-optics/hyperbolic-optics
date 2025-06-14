import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import json
from typing import Dict, Tuple, List, Optional, Union

# Configuration Constants
CONFIG = {
    'PLOT': {
        'FONT_FAMILY': 'Arial',
        'LABEL_SIZE': 18,
        'TICK_SIZE': 16,
        'TICK_WIDTH': 1.2,
        'TICK_LENGTH': 8,
        'TEXT_SIZE': 16,
        'DPI': 300,
        'COLORMAP': 'PuRd_r',
        'SAVE_PATH': Path('test_figures')
    },
    'PHYSICS': {
        'PRISM_PERMITTIVITY': 50.
    }
}

class PlotStyle:
    """Manages consistent plotting styles across all figures."""
    
    @staticmethod
    def initialize():
        """Initialize global matplotlib parameters."""
        plt.rcParams.update({
            'font.family': CONFIG['PLOT']['FONT_FAMILY'],
            'font.size': CONFIG['PLOT']['LABEL_SIZE'],
            'axes.labelsize': CONFIG['PLOT']['LABEL_SIZE'],
            'axes.titlesize': CONFIG['PLOT']['LABEL_SIZE'],
            'xtick.labelsize': CONFIG['PLOT']['TICK_SIZE'],
            'ytick.labelsize': CONFIG['PLOT']['TICK_SIZE'],
            'mathtext.fontset': 'custom',
            'mathtext.rm': CONFIG['PLOT']['FONT_FAMILY'],
            'mathtext.it': f"{CONFIG['PLOT']['FONT_FAMILY']}:italic",
            'mathtext.bf': f"{CONFIG['PLOT']['FONT_FAMILY']}:bold"
        })
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    @staticmethod
    def style_axis(ax: plt.Axes, show_labels: bool = True):
        """Apply consistent styling to axis."""
        ax.tick_params(
            width=CONFIG['PLOT']['TICK_WIDTH'],
            length=CONFIG['PLOT']['TICK_LENGTH'],
            direction='in',
            pad=5,
        )

def plot_permittivity(material, eps_ext, eps_ord, save_name: Optional[str] = None):
    """Plot permittivity spectra with improved styling."""
    PlotStyle.initialize()
    
    fig, axs = plt.subplots(2, figsize=(9, 7), sharex=True, 
                           gridspec_kw={'hspace': 0.1})
    
    # Plot real part
    axs[0].plot(material.frequency, tf.math.real(eps_ext),
                label=r"$\mathrm{Re}(\varepsilon_\mathrm{ext})$")
    axs[0].plot(material.frequency, tf.math.real(eps_ord),
                label=r"$\mathrm{Re}(\varepsilon_\mathrm{ord})$")
    axs[0].axhline(y=0, color="black", linewidth=1)
    axs[0].set(ylabel=r"$\mathrm{Re}(\epsilon)$")
    axs[0].legend()
    PlotStyle.style_axis(axs[0])
    
    # Plot imaginary part
    axs[1].plot(material.frequency, tf.math.imag(eps_ext),
                label=r"$\mathrm{Im}(\varepsilon_\mathrm{ext})$")
    axs[1].plot(material.frequency, tf.math.imag(eps_ord),
                label=r"$\mathrm{Im}(\varepsilon_\mathrm{ord})$")
    axs[1].set(xlabel=r"Wavenumber (cm$^{-1}$)", 
               ylabel=r"$\mathrm{Im}(\epsilon)$")
    axs[1].set_xlim(material.frequency[0].numpy(), 
                    material.frequency[-1].numpy())
    axs[1].set_ylim(0,)
    axs[1].legend()
    PlotStyle.style_axis(axs[1])
    
    if save_name:
        CONFIG['PLOT']['SAVE_PATH'].mkdir(exist_ok=True)
        plt.savefig(CONFIG['PLOT']['SAVE_PATH'] / f"{save_name}.png",
                   dpi=CONFIG['PLOT']['DPI'], bbox_inches="tight")
    plt.show()
    plt.close()


def plot_mueller_azimuthal(structure, param: np.ndarray,
                          title: Optional[str] = None,
                          save_name: Optional[str] = None,
                          label: str = "a"):
    """Plot frequency vs azimuthal angle with paper-quality styling.
    
    Args:
        structure: Structure object containing azimuthal angle and frequency data
        param: The parameter to plot
        title: Optional title for the plot
        save_name: Optional filename for saving the plot
        label: Optional subplot label (default: "a")
    """
    PlotStyle.initialize()
    
    # Create figure with gridspec for precise layout control
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=1,
        wspace=0.2,
        left=0.1,
        right=0.9,
        top=0.9,
        bottom=0.15
    )
    
    # Create axis with gridspec
    ax = fig.add_subplot(gs[0])
    
    # Calculate axis values
    x_axis = np.round(np.degrees(structure.azimuthal_angle.numpy().real), 1)
    frequency = structure.frequency.numpy().real
    
    # Create the color plot using pcolormesh
    im = ax.pcolormesh(x_axis, frequency, param,
                       cmap=CONFIG['PLOT']['COLORMAP'],
                       vmin=0,
                       vmax=1)
    
    # Set axis limits and ticks
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_ylim(frequency[0], frequency[-1])
    
    # Apply paper-quality tick styling
    ax.tick_params(
        labelsize=CONFIG['PLOT']['TICK_SIZE'],
        width=CONFIG['PLOT']['TICK_WIDTH'],
        length=CONFIG['PLOT']['TICK_LENGTH'],
        direction='in',
        pad=5,
        top=False,
        right=False
    )
    
    # Set axis labels with LaTeX formatting
    ax.set_xlabel(r"$\beta$ (degree)", 
                 fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                 labelpad=10)
    ax.set_ylabel(r"$\omega/2\pi c$ (cm$^{-1}$)",
                 fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                 labelpad=10)
    
    # Add subplot label in top left corner
    ax.text(0.06, 0.94, f'({label})',
            transform=ax.transAxes,
            fontsize=CONFIG['PLOT']['TEXT_SIZE'],
            va='top',
            ha='left')
    
    # Add title if provided
    if title:
        ax.text(0.5, 1.02, title,
                transform=ax.transAxes,
                fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                ha='center')
    
    # Set aspect ratio to make plot square
    ax.set_box_aspect(1)
    
    # Add custom positioned colorbar
    pos = ax.get_position()
    cbar_ax = fig.add_axes([
        pos.x1 + 0.01,
        pos.y0 + 0.12,
        0.01,
        pos.height * 0.8
    ])
    
    cbar = fig.colorbar(im, cax=cbar_ax, shrink=0.8, aspect=2, ticks=[0, 0.5, 1])
    cbar.set_label('Reflectance', size=16)
    cbar.ax.yaxis.set_tick_params(
        labelsize=14,
        width=0,
        length=0,
        direction='in',
        right=True,
        left=True,
        top=True
    )
    
    # Save the plot if a filename is provided
    if save_name:
        CONFIG['PLOT']['SAVE_PATH'].mkdir(exist_ok=True)
        plt.savefig(CONFIG['PLOT']['SAVE_PATH'] / f"{save_name}.png",
                   dpi=CONFIG['PLOT']['DPI'],
                   bbox_inches='tight',
                   pad_inches=0.1)
    
    plt.show()
    plt.close()

def plot_mueller_azimuthal_pair(structure, param1: np.ndarray, param2: np.ndarray,
                               title1: Optional[str] = None,
                               title2: Optional[str] = None,
                               save_name: Optional[str] = None):
    """Plot two azimuthal plots side by side with paper-quality styling.
    
    Args:
        structure: Structure object containing data
        param1: Parameter for first plot
        param2: Parameter for second plot
        title1: Optional title for first plot
        title2: Optional title for second plot
        save_name: Optional filename for saving the plot
    """
    PlotStyle.initialize()
    
    # Create figure with gridspec for precise layout control
    fig = plt.figure(figsize=CONFIG['PLOT']['FIGURE_SIZE'])
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1, 1],
        wspace=0.2,
        left=0.1,
        right=0.9,
        top=0.9,
        bottom=0.15
    )
    
    # Create first subplot
    ax1 = fig.add_subplot(gs[0])
    x_axis = np.round(np.degrees(structure.azimuthal_angle.numpy().real), 1)
    frequency = structure.frequency.numpy().real
    
    im1 = ax1.pcolormesh(x_axis, frequency, param1,
                         cmap=CONFIG['PLOT']['COLORMAP'],
                         vmin=0,
                         vmax=1)
    
    # Create second subplot
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    im2 = ax2.pcolormesh(x_axis, frequency, param2,
                         cmap=CONFIG['PLOT']['COLORMAP'],
                         vmin=0,
                         vmax=1)
    
    # Style both subplots
    for idx, (ax, title) in enumerate([(ax1, title1), (ax2, title2)]):
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_ylim(frequency[0], frequency[-1])
        
        ax.tick_params(
            labelsize=CONFIG['PLOT']['TICK_SIZE'],
            width=CONFIG['PLOT']['TICK_WIDTH'],
            length=CONFIG['PLOT']['TICK_LENGTH'],
            direction='in',
            pad=5,
            top=False,
            right=False
        )
        
        ax.set_xlabel(r"$\beta$ (degree)", 
                     fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                     labelpad=10)
        if idx == 0:  # Only add ylabel to first subplot
            ax.set_ylabel(r"$\omega/2\pi c$ (cm$^{-1}$)",
                         fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                         labelpad=10)
        else:
            ax.tick_params(labelleft=False)
        
        # Add subplot label
        ax.text(0.06, 0.94, f'({["a", "b"][idx]})',
                transform=ax.transAxes,
                fontsize=CONFIG['PLOT']['TEXT_SIZE'],
                va='top',
                ha='left')
        
        if title:
            ax.text(0.5, 1.02, title,
                    transform=ax.transAxes,
                    fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                    ha='center')
        
        ax.set_box_aspect(1)
    
    # Add colorbar to the right of the second subplot
    pos = ax2.get_position()
    cbar_ax = fig.add_axes([
        pos.x1 + 0.01,
        pos.y0 + 0.12,
        0.01,
        pos.height * 0.8
    ])
    
    cbar = fig.colorbar(im2, cax=cbar_ax, shrink=0.8, aspect=2, ticks=[0, 0.5, 1])
    cbar.set_label('Reflectance', size=16)
    cbar.ax.yaxis.set_tick_params(
        labelsize=14,
        width=0,
        length=0,
        direction='in',
        right=True,
        left=True,
        top=True
    )
    
    # Save the plot if a filename is provided
    if save_name:
        CONFIG['PLOT']['SAVE_PATH'].mkdir(exist_ok=True)
        plt.savefig(CONFIG['PLOT']['SAVE_PATH'] / f"{save_name}.png",
                   dpi=CONFIG['PLOT']['DPI'],
                   bbox_inches='tight',
                   pad_inches=0.1)
    
    plt.show()
    plt.close()


def plot_stokes_parameters(structure: object, params: Dict[str, np.ndarray],
                          plot_type: str = 'incidence',
                          save_name: Optional[str] = None):
    """Plot all Stokes parameters in a 2x3 grid."""
    PlotStyle.initialize()
    
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    
    ax_to_plot = [
        (params['S0'], "S0", 0, 0),
        (params['S1'], "S1", 0, 1),
        (params['S2'], "S2", 0, 2),
        (params['S3'], "S3", 1, 0),
        (params['DOP'], "DOP", 1, 1),
        (params['Ellipticity'], "Ellipticity", 1, 2),
    ]
    
    if plot_type == 'incidence':
        x_axis = np.round(np.degrees(structure.incident_angle.numpy().real), 1)
        xlabel = r"Incident Angle / $^\circ$"
    else:  # azimuthal
        x_axis = np.round(np.degrees(structure.azimuthal_angle.numpy().real), 1)
        xlabel = r"Azimuthal Rotation / $^\circ$"
    
    frequency = structure.frequency.numpy().real
    
    for data, title, row, col in ax_to_plot:
        im = ax[row, col].pcolormesh(x_axis, frequency, data,
                                    cmap=CONFIG['PLOT']['COLORMAP'])
        cbar = plt.colorbar(im, ax=ax[row, col])
        cbar.set_label(title, size=CONFIG['PLOT']['LABEL_SIZE'])
        ax[row, col].set_title(title, size=CONFIG['PLOT']['LABEL_SIZE'])
        ax[row, col].set_xlabel(xlabel)
        ax[row, col].set_ylabel(r"$\omega/2\pi c$ (cm$^{-1}$)")
        PlotStyle.style_axis(ax[row, col])
    
    plt.tight_layout()
    
    if save_name:
        CONFIG['PLOT']['SAVE_PATH'].mkdir(exist_ok=True)
        plt.savefig(CONFIG['PLOT']['SAVE_PATH'] / f"{save_name}.png",
                   dpi=CONFIG['PLOT']['DPI'], bbox_inches="tight")
    plt.show()
    plt.close()


def plot_kx_frequency(structure: object, param: np.ndarray,
                     title: Optional[str] = None,
                     rotation_y: Optional[float] = None,
                     save_name: Optional[str] = None,
                     label: str = "a"):
    """Plot frequency vs kx with paper-quality styling.
    
    Args:
        structure: Structure object containing eps_prism and frequency data
        param: The parameter to plot (e.g., reflectivity)
        title: Optional title for the plot
        rotation_y: Optional rotation angle to display
        save_name: Optional filename for saving the plot
        label: Optional subplot label (default: "a")
    """
    PlotStyle.initialize()
    
    # Create figure with gridspec for precise layout control
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=1,
        wspace=0.2,
        left=0.1,
        right=0.9,
        top=0.9,
        bottom=0.15
    )
    
    # Create axis with gridspec
    ax = fig.add_subplot(gs[0])
    
    # Calculate kx values from structure properties
    n_prism = np.sqrt(float(structure.eps_prism))
    incident_angles = structure.incident_angle.numpy().real
    kx = n_prism * np.sin(incident_angles)
    frequency = structure.frequency.numpy().real
    
    # Create the color plot
    im = ax.pcolormesh(kx, frequency, param,
                       cmap=CONFIG['PLOT']['COLORMAP'],
                       vmin=0,
                       vmax=1)
    
    # Set x-axis limits and generate ticks
    max_kx = n_prism
    ax.set_xlim(-max_kx, max_kx)
    
    # Determine step size based on the range
    if max_kx < 3:
        step = 1  # Half-integer steps for small ranges
    elif max_kx < 8:
        step = 2    # Integer steps for medium ranges
    elif max_kx < 15:
        step = 3    # Steps of 3 for larger ranges
    else:
        step = 5    # Steps of 5 for very large ranges
    
    # Calculate maximum tick value
    max_tick = (int(max_kx) // step) * step
    
    # Generate symmetrical ticks around zero
    positive_ticks = np.arange(0, max_tick + step/2, step)
    negative_ticks = -np.arange(step, max_tick + step/2, step)
    ticks = np.concatenate([negative_ticks, positive_ticks])
    ticks = ticks[np.abs(ticks) <= max_kx]
    ax.set_xticks(ticks)
    
    # Set y-axis limits
    ax.set_ylim(frequency[0], frequency[-1])
    
    # Apply paper-quality tick styling
    ax.tick_params(
        labelsize=CONFIG['PLOT']['TICK_SIZE'],
        width=CONFIG['PLOT']['TICK_WIDTH'],
        length=CONFIG['PLOT']['TICK_LENGTH'],
        direction='in',
        pad=5,
        top=False,
        right=False
    )
    
    # Set axis labels with LaTeX formatting
    ax.set_xlabel(r"$k_x/k_0$", 
                 fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                 labelpad=10)
    ax.set_ylabel(r"$\omega/2\pi c$ (cm$^{-1}$)",
                 fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                 labelpad=10)
    
    # Add subplot label in top left corner
    ax.text(0.06, 0.94, f'({label})',
            transform=ax.transAxes,
            fontsize=CONFIG['PLOT']['TEXT_SIZE'],
            va='top',
            ha='left')
    
    # Add rotation angle if provided
    if rotation_y is not None:
        ax.text(0.98, 0.96, 
                rf'$\varphi = {rotation_y}^{{\circ}}$',
                transform=ax.transAxes,
                fontsize=CONFIG['PLOT']['TEXT_SIZE'],
                ha='right',
                va='top')
    
    # Add title if provided
    if title:
        ax.text(0.5, 1.02, title,
                transform=ax.transAxes,
                fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                ha='center')
    
    # Set aspect ratio to make plot square
    ax.set_box_aspect(1)
    
    # Add custom positioned colorbar
    pos = ax.get_position()
    cbar_ax = fig.add_axes([
        pos.x1 + 0.01,
        pos.y0 + 0.12,
        0.01,
        pos.height * 0.8
    ])
    
    cbar = fig.colorbar(im, cax=cbar_ax, shrink=0.8, aspect=2, ticks=[0, 0.5, 1])
    cbar.set_label('Reflectance', size=16)
    cbar.ax.yaxis.set_tick_params(
        labelsize=14,
        width=0,
        length=0,
        direction='in',
        right=True,
        left=True,
        top=True
    )
    
    # Save plot if filename provided
    if save_name:
        CONFIG['PLOT']['SAVE_PATH'].mkdir(exist_ok=True)
        plt.savefig(CONFIG['PLOT']['SAVE_PATH'] / f"{save_name}.png",
                   dpi=CONFIG['PLOT']['DPI'],
                   bbox_inches='tight',
                   pad_inches=0.1)
    
    plt.show()
    plt.close()

def plot_kx_frequency_pair(structure, param1: np.ndarray, param2: np.ndarray,
                          rotation_y1: Optional[float] = None,
                          rotation_y2: Optional[float] = None,
                          title1: Optional[str] = None,
                          title2: Optional[str] = None,
                          save_name: Optional[str] = None):
    """Plot two kx-frequency plots side by side with paper-quality styling.
    
    Args:
        structure: Structure object containing data
        param1: Parameter for first plot
        param2: Parameter for second plot
        rotation_y1: Optional rotation angle for first plot
        rotation_y2: Optional rotation angle for second plot
        title1: Optional title for first plot
        title2: Optional title for second plot
        save_name: Optional filename for saving the plot
    """
    PlotStyle.initialize()
    
    # Create figure with gridspec for precise layout control
    fig = plt.figure(figsize=CONFIG['PLOT']['FIGURE_SIZE'])
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1, 1],
        wspace=0.2,
        left=0.1,
        right=0.9,
        top=0.9,
        bottom=0.15
    )
    
    # Calculate common data
    n_prism = np.sqrt(float(structure.eps_prism))
    incident_angles = structure.incident_angle.numpy().real
    kx = n_prism * np.sin(incident_angles)
    frequency = structure.frequency.numpy().real
    max_kx = n_prism
    
    # Create first subplot
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.pcolormesh(kx, frequency, param1,
                         cmap=CONFIG['PLOT']['COLORMAP'],
                         vmin=0,
                         vmax=1)
    
    # Create second subplot
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    im2 = ax2.pcolormesh(kx, frequency, param2,
                         cmap=CONFIG['PLOT']['COLORMAP'],
                         vmin=0,
                         vmax=1)
    
    # Style both subplots
    rotation_angles = [rotation_y1, rotation_y2]
    for idx, (ax, title, rot_y) in enumerate([(ax1, title1, rotation_y1),
                                             (ax2, title2, rotation_y2)]):
        # Set limits and generate ticks
        ax.set_xlim(-max_kx, max_kx)
        
        # Determine step size
        if max_kx < 3:
            step = 0.5
        elif max_kx < 8:
            step = 1
        elif max_kx < 15:
            step = 3
        else:
            step = 5
        
        max_tick = (int(max_kx) // step) * step
        positive_ticks = np.arange(0, max_tick + step/2, step)
        negative_ticks = -np.arange(step, max_tick + step/2, step)
        ticks = np.concatenate([negative_ticks, positive_ticks])
        ticks = ticks[np.abs(ticks) <= max_kx]
        ax.set_xticks(ticks)
        
        ax.set_ylim(frequency[0], frequency[-1])
        
        # Apply tick styling
        ax.tick_params(
            labelsize=CONFIG['PLOT']['TICK_SIZE'],
            width=CONFIG['PLOT']['TICK_WIDTH'],
            length=CONFIG['PLOT']['TICK_LENGTH'],
            direction='in',
            pad=5,
            top=False,
            right=False
        )
        
        # Set labels
        ax.set_xlabel(r"$k_x/k_0$", 
                     fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                     labelpad=10)
        if idx == 0:  # Only add ylabel to first subplot
            ax.set_ylabel(r"$\omega/2\pi c$ (cm$^{-1}$)",
                         fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                         labelpad=10)
        else:
            ax.tick_params(labelleft=False)
        
        # Add subplot label
        ax.text(0.06, 0.94, f'({["a", "b"][idx]})',
                transform=ax.transAxes,
                fontsize=CONFIG['PLOT']['TEXT_SIZE'],
                va='top',
                ha='left')
        
        # Add rotation angle if provided
        if rot_y is not None:
            ax.text(0.98, 0.96, 
                    rf'$\varphi = {rot_y}^{{\circ}}$',
                    transform=ax.transAxes,
                    fontsize=CONFIG['PLOT']['TEXT_SIZE'],
                    ha='right',
                    va='top')
        
        if title:
            ax.text(0.5, 1.02, title,
                    transform=ax.transAxes,
                    fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                    ha='center')
        
        ax.set_box_aspect(1)
    
    # Add colorbar to the right of the second subplot
    pos = ax2.get_position()
    cbar_ax = fig.add_axes([
        pos.x1 + 0.01,
        pos.y0 + 0.12,
        0.01,
        pos.height * 0.8
    ])
    
    cbar = fig.colorbar(im2, cax=cbar_ax, shrink=0.8, aspect=2, ticks=[0, 0.5, 1])
    cbar.set_label('Reflectance', size=16)
    cbar.ax.yaxis.set_tick_params(
        labelsize=14,
        width=0,
        length=0,
        direction='in',
        right=True,
        left=True,
        top=True
    )
    
    # Save plot if filename provided
    if save_name:
        CONFIG['PLOT']['SAVE_PATH'].mkdir(exist_ok=True)
        plt.savefig(CONFIG['PLOT']['SAVE_PATH'] / f"{save_name}.png",
                   dpi=CONFIG['PLOT']['DPI'],
                   bbox_inches='tight',
                   pad_inches=0.1)
    
    plt.show()
    plt.close()


def plot_mueller_dispersion(structure: object, param: np.ndarray,
                          title: Optional[str] = None,
                          rotation_y: Optional[float] = None,
                          save_name: Optional[str] = None,
                          label: str = "a"):
    """Plot k-space dispersion with paper-quality styling in kx-ky coordinates.
    
    Args:
        structure: Structure object containing angle and frequency data
        param: The parameter to plot
        title: Optional title for the plot
        rotation_y: Optional rotation angle to display
        save_name: Optional filename for saving the plot
        label: Optional subplot label (default: "a")
    """
    PlotStyle.initialize()
    
    # Create figure with gridspec for precise layout control
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=1,
        wspace=0.2,
        left=0.1,
        right=0.9,
        top=0.9,
        bottom=0.15
    )
    
    # Create axis with gridspec
    ax = fig.add_subplot(gs[0])
    
    # Calculate k-space coordinates
    incident_angle = structure.incident_angle.numpy().real
    z_rotation = structure.azimuthal_angle.numpy().real
    max_k = np.sqrt(float(structure.eps_prism))  # Maximum k value from prism
    
    # Create meshgrid for incident angle and z-rotation
    incident_angle, z_rotation = np.meshgrid(incident_angle, z_rotation)
    
    # Convert polar coordinates to Cartesian (kx, ky)
    kx = max_k * np.sin(incident_angle) * np.cos(z_rotation)
    ky = max_k * np.sin(incident_angle) * np.sin(z_rotation)
    
    # Create the color plot
    im = ax.pcolormesh(kx, ky, param.numpy().T,
                       cmap=CONFIG['PLOT']['COLORMAP'],
                       vmin=0,
                       vmax=1)
    
    # Set plot limits and aspect ratio
    ax.set_aspect('equal')
    ax.set_xlim(-max_k * 1.1, max_k * 1.1)
    ax.set_ylim(-max_k * 1.1, max_k * 1.1)

    # Determine step size based on the range (same logic as plot_kx_frequency)
    if max_k < 3:
        step = 1  # Integer steps for small ranges
    elif max_k < 8:
        step = 2    # Steps of 2 for medium ranges
    elif max_k < 15:
        step = 3    # Steps of 3 for larger ranges
    else:
        step = 5    # Steps of 5 for very large ranges
    
    # Calculate maximum tick value
    max_tick = (int(max_k) // step) * step
    
    # Generate symmetrical ticks around zero
    positive_ticks = np.arange(0, max_tick + step/2, step)
    negative_ticks = -np.arange(step, max_tick + step/2, step)
    ticks = np.concatenate([negative_ticks, positive_ticks])
    ticks = ticks[np.abs(ticks) <= max_k]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    # Apply paper-quality tick styling
    ax.tick_params(
        labelsize=CONFIG['PLOT']['TICK_SIZE'],
        width=CONFIG['PLOT']['TICK_WIDTH'],
        length=CONFIG['PLOT']['TICK_LENGTH'],
        direction='in',
        pad=5,
        top=False,
        right=False
    )
    
    # Set axis labels with LaTeX formatting
    ax.set_xlabel(r"$k_x/k_0$", 
                 fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                 labelpad=10)
    ax.set_ylabel(r"$k_y/k_0$",
                 fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                 labelpad=10)
    
    # Add subplot label in top left corner
    ax.text(0.06, 0.94, f'({label})',
            transform=ax.transAxes,
            fontsize=CONFIG['PLOT']['TEXT_SIZE'],
            va='top',
            ha='left')
    
    # Add rotation angle if provided
    if rotation_y is not None:
        ax.text(0.98, 0.96, 
                rf'$\varphi = {rotation_y}^{{\circ}}$',
                transform=ax.transAxes,
                fontsize=CONFIG['PLOT']['TEXT_SIZE'],
                ha='right',
                va='top')
    
    # Add title if provided
    if title:
        ax.text(0.5, 1.02, title,
                transform=ax.transAxes,
                fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                ha='center')
    
    # Add unit circle to indicate light cone
    circle = plt.Circle((0, 0), 1, fill=False,
                       color='white', linestyle='-', linewidth=1.5)
    ax.add_patch(circle)
    
    # Set aspect ratio to make plot square
    ax.set_box_aspect(1)
    
    # Add custom positioned colorbar
    pos = ax.get_position()
    cbar_ax = fig.add_axes([
        pos.x1 + 0.01,
        pos.y0 + 0.12,
        0.01,
        pos.height * 0.8
    ])
    
    cbar = fig.colorbar(im, cax=cbar_ax, shrink=0.8, aspect=2, ticks=[0, 0.5, 1])
    cbar.set_label('Reflectance', size=16)
    cbar.ax.yaxis.set_tick_params(
        labelsize=14,
        width=0,
        length=0,
        direction='in',
        right=True,
        left=True,
        top=True
    )
    
    # Save plot if filename provided
    if save_name:
        CONFIG['PLOT']['SAVE_PATH'].mkdir(exist_ok=True)
        plt.savefig(CONFIG['PLOT']['SAVE_PATH'] / f"{save_name}.png",
                   dpi=CONFIG['PLOT']['DPI'],
                   bbox_inches='tight',
                   pad_inches=0.1)
    
    plt.show()
    plt.close()

def plot_mueller_dispersion_pair(structure, param1: np.ndarray, param2: np.ndarray,
                                rotation_y1: Optional[float] = None,
                                rotation_y2: Optional[float] = None,
                                title1: Optional[str] = None,
                                title2: Optional[str] = None,
                                save_name: Optional[str] = None):
    """Plot two k-space dispersion plots side by side with paper-quality styling.
    
    Args:
        structure: Structure object containing data
        param1: Parameter for first plot
        param2: Parameter for second plot
        rotation_y1: Optional rotation angle for first plot
        rotation_y2: Optional rotation angle for second plot
        title1: Optional title for first plot
        title2: Optional title for second plot
        save_name: Optional filename for saving the plot
    """
    PlotStyle.initialize()
    
    # Create figure with gridspec for precise layout control
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1, 1],
        wspace=0.2,
        left=0.1,
        right=0.9,
        top=0.9,
        bottom=0.15
    )
    
    # Calculate common k-space coordinates
    incident_angle = structure.incident_angle.numpy().real
    z_rotation = structure.azimuthal_angle.numpy().real
    max_k = np.sqrt(float(structure.eps_prism))
    
    incident_angle, z_rotation = np.meshgrid(incident_angle, z_rotation)
    kx = max_k * np.sin(incident_angle) * np.cos(z_rotation)
    ky = max_k * np.sin(incident_angle) * np.sin(z_rotation)
    
    # Create first subplot
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.pcolormesh(kx, ky, param1.numpy().T,
                         cmap=CONFIG['PLOT']['COLORMAP'],
                         vmin=0,
                         vmax=1)
    
    # Create second subplot
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    im2 = ax2.pcolormesh(kx, ky, param2.numpy().T,
                         cmap=CONFIG['PLOT']['COLORMAP'],
                         vmin=0,
                         vmax=1)
    
    # Style both subplots
    for idx, (ax, title, rot_y) in enumerate([(ax1, title1, rotation_y1),
                                             (ax2, title2, rotation_y2)]):
        # Set plot limits and aspect ratio
        ax.set_aspect('equal')
        ax.set_xlim(-max_k * 1.05, max_k * 1.05)
        ax.set_ylim(-max_k * 1.05, max_k * 1.05)
        
        # Set ticks based on max_k
        if max_k < 3:
            tick_spacing = 1
        elif max_k < 6:
            tick_spacing = 2
        else:
            tick_spacing = 3

        # Generate negative ticks (going backwards from 0)
        neg_ticks = np.arange(0, -int(max_k) - 1, -tick_spacing)
        # Generate positive ticks (going forwards from 0)
        pos_ticks = np.arange(0, int(max_k) + 1, tick_spacing)
        # Combine them, excluding the duplicate 0
        ticks = np.concatenate([neg_ticks[1:], pos_ticks])

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        
        # Apply tick styling
        ax.tick_params(
            labelsize=CONFIG['PLOT']['TICK_SIZE'],
            width=CONFIG['PLOT']['TICK_WIDTH'],
            length=CONFIG['PLOT']['TICK_LENGTH'],
            direction='in',
            pad=5,
            top=False,
            right=False
        )
        
        # Set labels
        ax.set_xlabel(r"$k_x/k_0$", 
                     fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                     labelpad=10)
        if idx == 0:  # Only add ylabel to first subplot
            ax.set_ylabel(r"$k_y/k_0$",
                         fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                         labelpad=10)
        else:
            ax.tick_params(labelleft=False)
        
        # Add subplot label
        ax.text(0.06, 0.94, f'({["a", "b"][idx]})',
                transform=ax.transAxes,
                fontsize=CONFIG['PLOT']['TEXT_SIZE'],
                va='top',
                ha='left')
        
        # Add rotation angle if provided
        if rot_y is not None:
            ax.text(0.98, 0.96, 
                    rf'$\varphi = {rot_y}^{{\circ}}$',
                    transform=ax.transAxes,
                    fontsize=CONFIG['PLOT']['TEXT_SIZE'],
                    ha='right',
                    va='top')
        
        if title:
            ax.text(0.5, 1.02, title,
                    transform=ax.transAxes,
                    fontsize=CONFIG['PLOT']['LABEL_SIZE'],
                    ha='center')
        
        # Add unit circle
        circle = plt.Circle((0, 0), 1, fill=False,
                          color='white', linestyle='-', linewidth=1.5)
        ax.add_patch(circle)
        
        ax.set_box_aspect(1)
    
    # Add colorbar to the right of the second subplot
    pos = ax2.get_position()
    cbar_ax = fig.add_axes([
        pos.x1 + 0.01,
        pos.y0 + 0.12,
        0.01,
        pos.height * 0.8
    ])
    
    cbar = fig.colorbar(im2, cax=cbar_ax, shrink=0.8, aspect=2, ticks=[0, 0.5, 1])
    cbar.set_label('Reflectance', size=16)
    cbar.ax.yaxis.set_tick_params(
        labelsize=14,
        width=0,
        length=0,
        direction='in',
        right=True,
        left=True,
        top=True
    )
    
    # Save plot if filename provided
    if save_name:
        CONFIG['PLOT']['SAVE_PATH'].mkdir(exist_ok=True)
        plt.savefig(CONFIG['PLOT']['SAVE_PATH'] / f"{save_name}.png",
                   dpi=CONFIG['PLOT']['DPI'],
                   bbox_inches='tight',
                   pad_inches=0.1)
    
    plt.show()
    plt.close()