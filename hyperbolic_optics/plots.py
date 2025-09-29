"""Publication-quality plotting functions for simulation results.

This module provides plotting utilities for various visualization types:

    - Material permittivity spectra
    - Frequency vs incident angle (kx-ω diagrams)
    - Frequency vs azimuthal angle (β-ω diagrams)
    - k-space dispersion (kx-ky isofrequency contours)
    - Stokes parameter distributions
    - Mueller matrix elements

All plots use consistent styling for publication-ready figures with
proper axis labels, colorbars, and typography.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hyperbolic_optics.materials import BaseMaterial
from hyperbolic_optics.structure import Structure

# Configuration Constants
CONFIG = {
    "PLOT": {
        "FONT_FAMILY": "Arial",
        "LABEL_SIZE": 18,
        "TICK_SIZE": 16,
        "TICK_WIDTH": 1.2,
        "TICK_LENGTH": 8,
        "TEXT_SIZE": 16,
        "DPI": 300,
        "COLORMAP": "PuRd_r",
        "SAVE_PATH": Path("test_figures"),
        "FIGURE_SIZE": (20, 5),  # ADD THIS
    },
    "PHYSICS": {"PRISM_PERMITTIVITY": 50.0},
}


class PlotStyle:
    """Manages consistent plotting styles across all figures."""

    @staticmethod
    def initialize() -> None:
        """Initialize global matplotlib parameters for consistent plotting style.

        Sets font family, sizes, tick parameters, and math text rendering
        for publication-quality plots.
        """
        plt.rcParams.update(
            {
                "font.family": CONFIG["PLOT"]["FONT_FAMILY"],
                "font.size": CONFIG["PLOT"]["LABEL_SIZE"],
                "axes.labelsize": CONFIG["PLOT"]["LABEL_SIZE"],
                "axes.titlesize": CONFIG["PLOT"]["LABEL_SIZE"],
                "xtick.labelsize": CONFIG["PLOT"]["TICK_SIZE"],
                "ytick.labelsize": CONFIG["PLOT"]["TICK_SIZE"],
                "mathtext.fontset": "custom",
                "mathtext.rm": CONFIG["PLOT"]["FONT_FAMILY"],
                "mathtext.it": f"{CONFIG['PLOT']['FONT_FAMILY']}:italic",
                "mathtext.bf": f"{CONFIG['PLOT']['FONT_FAMILY']}:bold",
            }
        )

    @staticmethod
    def style_axis(ax: plt.Axes, show_labels: bool = True) -> None:
        """Apply consistent styling to matplotlib axis.

        Args:
            ax: Matplotlib axes object to style
            show_labels: Whether to show axis labels

        Note:
            Sets tick width, length, direction, and padding.
        """
        ax.tick_params(
            width=CONFIG["PLOT"]["TICK_WIDTH"],
            length=CONFIG["PLOT"]["TICK_LENGTH"],
            direction="in",
            pad=5,
        )


def plot_permittivity(
    material: BaseMaterial, eps_ext: np.ndarray, eps_ord: np.ndarray, save_name: str | None = None
) -> None:
    """Plot real and imaginary parts of permittivity spectra.

    Args:
        material: Material object containing frequency array
        eps_ext: Extraordinary permittivity (parallel to optical axis)
        eps_ord: Ordinary permittivity (perpendicular to optical axis)
        save_name: Optional filename for saving (without extension)

    Note:
        Creates two-panel plot with Re(ε) and Im(ε) vs frequency.
    """
    PlotStyle.initialize()

    fig, axs = plt.subplots(2, figsize=(9, 7), sharex=True, gridspec_kw={"hspace": 0.1})

    # Plot real part
    axs[0].plot(
        material.frequency,
        eps_ext.real,
        label=r"$\mathrm{Re}(\varepsilon_\mathrm{ext})$",
    )
    axs[0].plot(
        material.frequency,
        eps_ord.real,
        label=r"$\mathrm{Re}(\varepsilon_\mathrm{ord})$",
    )
    axs[0].axhline(y=0, color="black", linewidth=1)
    axs[0].set(ylabel=r"$\mathrm{Re}(\epsilon)$")
    axs[0].legend()
    PlotStyle.style_axis(axs[0])

    # Plot imaginary part
    axs[1].plot(
        material.frequency,
        eps_ext.imag,
        label=r"$\mathrm{Im}(\varepsilon_\mathrm{ext})$",
    )
    axs[1].plot(
        material.frequency,
        eps_ord.imag,
        label=r"$\mathrm{Im}(\varepsilon_\mathrm{ord})$",
    )
    axs[1].set(xlabel=r"Wavenumber (cm$^{-1}$)", ylabel=r"$\mathrm{Im}(\epsilon)$")
    axs[1].set_xlim(material.frequency[0], material.frequency[-1])
    axs[1].set_ylim(
        0,
    )
    axs[1].legend()
    PlotStyle.style_axis(axs[1])

    if save_name:
        CONFIG["PLOT"]["SAVE_PATH"].mkdir(exist_ok=True)
        plt.savefig(
            CONFIG["PLOT"]["SAVE_PATH"] / f"{save_name}.png",
            dpi=CONFIG["PLOT"]["DPI"],
            bbox_inches="tight",
        )
    plt.show()
    plt.close()


def plot_mueller_azimuthal(
    structure: Structure,
    param: np.ndarray,
    title: str | None = None,
    save_name: str | None = None,
    label: str = "a",
) -> None:
    """Plot frequency vs azimuthal angle with publication styling.

    Args:
        structure: Structure object with azimuthal_angle and frequency arrays
        param: 2D parameter array to plot (typically reflectivity) [410, 360]
        title: Optional plot title
        save_name: Optional filename for saving (without extension)
        label: Subplot label (e.g., 'a', 'b')

    Note:
        Creates color plot with frequency on y-axis and azimuthal angle
        (β) on x-axis, suitable for studying rotational anisotropy.
    """
    PlotStyle.initialize()

    # Create figure with gridspec for precise layout control
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(nrows=1, ncols=1, wspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.15)

    # Create axis with gridspec
    ax = fig.add_subplot(gs[0])

    # Calculate axis values
    x_axis = np.round(np.degrees(structure.azimuthal_angle), 1)
    frequency = structure.frequency

    # Create the color plot using pcolormesh
    im = ax.pcolormesh(x_axis, frequency, param, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)

    # Set axis limits and ticks
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_ylim(frequency[0], frequency[-1])

    # Apply paper-quality tick styling
    ax.tick_params(
        labelsize=CONFIG["PLOT"]["TICK_SIZE"],
        width=CONFIG["PLOT"]["TICK_WIDTH"],
        length=CONFIG["PLOT"]["TICK_LENGTH"],
        direction="in",
        pad=5,
        top=False,
        right=False,
    )

    # Set axis labels with LaTeX formatting
    ax.set_xlabel(r"$\beta$ (degree)", fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
    ax.set_ylabel(
        r"$\omega/2\pi c$ (cm$^{-1}$)",
        fontsize=CONFIG["PLOT"]["LABEL_SIZE"],
        labelpad=10,
    )

    # Add subplot label in top left corner
    ax.text(
        0.06,
        0.94,
        f"({label})",
        transform=ax.transAxes,
        fontsize=CONFIG["PLOT"]["TEXT_SIZE"],
        va="top",
        ha="left",
    )

    # Add title if provided
    if title:
        ax.text(
            0.5,
            1.02,
            title,
            transform=ax.transAxes,
            fontsize=CONFIG["PLOT"]["LABEL_SIZE"],
            ha="center",
        )

    # Set aspect ratio to make plot square
    ax.set_box_aspect(1)

    # Add custom positioned colorbar
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0 + 0.12, 0.01, pos.height * 0.8])

    cbar = fig.colorbar(im, cax=cbar_ax, shrink=0.8, aspect=2, ticks=[0, 0.5, 1])
    cbar.set_label("Reflectance", size=16)
    cbar.ax.yaxis.set_tick_params(
        labelsize=14, width=0, length=0, direction="in", right=True, left=True, top=True
    )

    # Save the plot if a filename is provided
    if save_name:
        CONFIG["PLOT"]["SAVE_PATH"].mkdir(exist_ok=True)
        plt.savefig(
            CONFIG["PLOT"]["SAVE_PATH"] / f"{save_name}.png",
            dpi=CONFIG["PLOT"]["DPI"],
            bbox_inches="tight",
            pad_inches=0.1,
        )

    plt.show()
    plt.close()


def plot_mueller_azimuthal_pair(
    structure: Structure,
    param1: np.ndarray,
    param2: np.ndarray,
    title1: str | None = None,
    title2: str | None = None,
    save_name: str | None = None,
) -> None:
    """Plot two azimuthal plots side-by-side for comparison.

    Args:
        structure: Structure object with angle and frequency data
        param1: First parameter array [410, 360]
        param2: Second parameter array [410, 360]
        title1: Title for left panel
        title2: Title for right panel
        save_name: Optional filename for saving

    Note:
        Useful for comparing different polarizations or materials.
    """
    PlotStyle.initialize()

    # Create figure with gridspec for precise layout control
    fig = plt.figure(figsize=CONFIG["PLOT"]["FIGURE_SIZE"])
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1, 1],
        wspace=0.2,
        left=0.1,
        right=0.9,
        top=0.9,
        bottom=0.15,
    )

    # Create first subplot
    ax1 = fig.add_subplot(gs[0])
    x_axis = np.round(np.degrees(structure.azimuthal_angle), 1)
    frequency = structure.frequency

    ax1.pcolormesh(x_axis, frequency, param1, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)

    # Create second subplot
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    im2 = ax2.pcolormesh(x_axis, frequency, param2, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)

    # Style both subplots
    for idx, (ax, title) in enumerate([(ax1, title1), (ax2, title2)]):
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_ylim(frequency[0], frequency[-1])

        ax.tick_params(
            labelsize=CONFIG["PLOT"]["TICK_SIZE"],
            width=CONFIG["PLOT"]["TICK_WIDTH"],
            length=CONFIG["PLOT"]["TICK_LENGTH"],
            direction="in",
            pad=5,
            top=False,
            right=False,
        )

        ax.set_xlabel(r"$\beta$ (degree)", fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
        if idx == 0:  # Only add ylabel to first subplot
            ax.set_ylabel(
                r"$\omega/2\pi c$ (cm$^{-1}$)",
                fontsize=CONFIG["PLOT"]["LABEL_SIZE"],
                labelpad=10,
            )
        else:
            ax.tick_params(labelleft=False)

        # Add subplot label
        ax.text(
            0.06,
            0.94,
            f'({["a", "b"][idx]})',
            transform=ax.transAxes,
            fontsize=CONFIG["PLOT"]["TEXT_SIZE"],
            va="top",
            ha="left",
        )

        if title:
            ax.text(
                0.5,
                1.02,
                title,
                transform=ax.transAxes,
                fontsize=CONFIG["PLOT"]["LABEL_SIZE"],
                ha="center",
            )

        ax.set_box_aspect(1)

    # Add colorbar to the right of the second subplot
    pos = ax2.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0 + 0.12, 0.01, pos.height * 0.8])

    cbar = fig.colorbar(im2, cax=cbar_ax, shrink=0.8, aspect=2, ticks=[0, 0.5, 1])
    cbar.set_label("Reflectance", size=16)
    cbar.ax.yaxis.set_tick_params(
        labelsize=14, width=0, length=0, direction="in", right=True, left=True, top=True
    )

    # Save the plot if a filename is provided
    if save_name:
        CONFIG["PLOT"]["SAVE_PATH"].mkdir(exist_ok=True)
        plt.savefig(
            CONFIG["PLOT"]["SAVE_PATH"] / f"{save_name}.png",
            dpi=CONFIG["PLOT"]["DPI"],
            bbox_inches="tight",
            pad_inches=0.1,
        )

    plt.show()
    plt.close()


def plot_stokes_parameters(
    structure: Structure,
    params: dict[str, np.ndarray],
    plot_type: str = "incidence",
    save_name: str | None = None,
) -> None:
    """Plot all Stokes parameters and DOP in 2×3 grid.

    Args:
        structure: Structure object with angle and frequency arrays
        params: Dictionary with S0, S1, S2, S3, DOP, Ellipticity keys
        plot_type: 'incidence' or 'azimuthal' to determine x-axis
        save_name: Optional filename for saving

    Note:
        Creates comprehensive visualization of complete polarization state
        across parameter space.
    """
    PlotStyle.initialize()

    fig, ax = plt.subplots(2, 3, figsize=(18, 12))

    ax_to_plot = [
        (params["S0"], "S0", 0, 0),
        (params["S1"], "S1", 0, 1),
        (params["S2"], "S2", 0, 2),
        (params["S3"], "S3", 1, 0),
        (params["DOP"], "DOP", 1, 1),
        (params["Ellipticity"], "Ellipticity", 1, 2),
    ]

    if plot_type == "incidence":
        x_axis = np.round(np.degrees(structure.incident_angle), 1)
        xlabel = r"Incident Angle / $^\circ$"
    else:  # azimuthal
        x_axis = np.round(np.degrees(structure.azimuthal_angle), 1)
        xlabel = r"Azimuthal Rotation / $^\circ$"

    frequency = structure.frequency

    for data, title, row, col in ax_to_plot:
        im = ax[row, col].pcolormesh(x_axis, frequency, data, cmap=CONFIG["PLOT"]["COLORMAP"])
        cbar = plt.colorbar(im, ax=ax[row, col])
        cbar.set_label(title, size=CONFIG["PLOT"]["LABEL_SIZE"])
        ax[row, col].set_title(title, size=CONFIG["PLOT"]["LABEL_SIZE"])
        ax[row, col].set_xlabel(xlabel)
        ax[row, col].set_ylabel(r"$\omega/2\pi c$ (cm$^{-1}$)")
        PlotStyle.style_axis(ax[row, col])

    plt.tight_layout()

    if save_name:
        CONFIG["PLOT"]["SAVE_PATH"].mkdir(exist_ok=True)
        plt.savefig(
            CONFIG["PLOT"]["SAVE_PATH"] / f"{save_name}.png",
            dpi=CONFIG["PLOT"]["DPI"],
            bbox_inches="tight",
        )
    plt.show()
    plt.close()


def plot_kx_frequency(
    structure: Structure,
    param: np.ndarray,
    title: str | None = None,
    rotation_y: float | None = None,
    save_name: str | None = None,
    label: str = "a",
) -> None:
    """Plot frequency vs parallel wavevector (kx) dispersion diagram.

    Args:
        structure: Structure object with incident_angle, frequency, eps_prism
        param: 2D parameter array [410, 360]
        title: Optional plot title
        rotation_y: Optional rotation angle to display in plot
        save_name: Optional filename for saving
        label: Subplot label

    Note:
        Shows polariton dispersion with kx/k0 on x-axis and frequency on
        y-axis. Useful for identifying resonance branches.
    """
    PlotStyle.initialize()

    # Create figure with gridspec for precise layout control
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(nrows=1, ncols=1, wspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.15)

    # Create axis with gridspec
    ax = fig.add_subplot(gs[0])

    # Calculate kx values from structure properties
    n_prism = np.sqrt(float(structure.eps_prism))
    incident_angles = structure.incident_angle
    kx = n_prism * np.sin(incident_angles)
    frequency = structure.frequency

    # Create the color plot
    im = ax.pcolormesh(kx, frequency, param, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)

    # Set x-axis limits and generate ticks
    max_kx = n_prism
    ax.set_xlim(-max_kx, max_kx)

    # Determine step size based on the range
    if max_kx < 3:
        step = 1  # Half-integer steps for small ranges
    elif max_kx < 8:
        step = 2  # Integer steps for medium ranges
    elif max_kx < 15:
        step = 3  # Steps of 3 for larger ranges
    else:
        step = 5  # Steps of 5 for very large ranges

    # Calculate maximum tick value
    max_tick = (int(max_kx) // step) * step

    # Generate symmetrical ticks around zero
    positive_ticks = np.arange(0, max_tick + step / 2, step)
    negative_ticks = -np.arange(step, max_tick + step / 2, step)
    ticks = np.concatenate([negative_ticks, positive_ticks])
    ticks = ticks[np.abs(ticks) <= max_kx]
    ax.set_xticks(ticks)

    # Set y-axis limits
    ax.set_ylim(frequency[0], frequency[-1])

    # Apply paper-quality tick styling
    ax.tick_params(
        labelsize=CONFIG["PLOT"]["TICK_SIZE"],
        width=CONFIG["PLOT"]["TICK_WIDTH"],
        length=CONFIG["PLOT"]["TICK_LENGTH"],
        direction="in",
        pad=5,
        top=False,
        right=False,
    )

    # Set axis labels with LaTeX formatting
    ax.set_xlabel(r"$k_x/k_0$", fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
    ax.set_ylabel(
        r"$\omega/2\pi c$ (cm$^{-1}$)",
        fontsize=CONFIG["PLOT"]["LABEL_SIZE"],
        labelpad=10,
    )

    # Add subplot label in top left corner
    ax.text(
        0.06,
        0.94,
        f"({label})",
        transform=ax.transAxes,
        fontsize=CONFIG["PLOT"]["TEXT_SIZE"],
        va="top",
        ha="left",
    )

    # Add rotation angle if provided
    if rotation_y is not None:
        ax.text(
            0.98,
            0.96,
            rf"$\varphi = {rotation_y}^{{\circ}}$",
            transform=ax.transAxes,
            fontsize=CONFIG["PLOT"]["TEXT_SIZE"],
            ha="right",
            va="top",
        )

    # Add title if provided
    if title:
        ax.text(
            0.5,
            1.02,
            title,
            transform=ax.transAxes,
            fontsize=CONFIG["PLOT"]["LABEL_SIZE"],
            ha="center",
        )

    # Set aspect ratio to make plot square
    ax.set_box_aspect(1)

    # Add custom positioned colorbar
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0 + 0.12, 0.01, pos.height * 0.8])

    cbar = fig.colorbar(im, cax=cbar_ax, shrink=0.8, aspect=2, ticks=[0, 0.5, 1])
    cbar.set_label("Reflectance", size=16)
    cbar.ax.yaxis.set_tick_params(
        labelsize=14, width=0, length=0, direction="in", right=True, left=True, top=True
    )

    # Save plot if filename provided
    if save_name:
        CONFIG["PLOT"]["SAVE_PATH"].mkdir(exist_ok=True)
        plt.savefig(
            CONFIG["PLOT"]["SAVE_PATH"] / f"{save_name}.png",
            dpi=CONFIG["PLOT"]["DPI"],
            bbox_inches="tight",
            pad_inches=0.1,
        )

    plt.show()
    plt.close()


def plot_kx_frequency_pair(
    structure: Structure,
    param1: np.ndarray,
    param2: np.ndarray,
    rotation_y1: float | None = None,
    rotation_y2: float | None = None,
    title1: str | None = None,
    title2: str | None = None,
    save_name: str | None = None,
) -> None:
    """Plot two kx-frequency diagrams side-by-side.

    Args:
        structure: Structure object with dispersion data
        param1: First parameter array
        param2: Second parameter array
        rotation_y1: Rotation angle for first plot
        rotation_y2: Rotation angle for second plot
        title1: Title for left panel
        title2: Title for right panel
        save_name: Optional filename for saving
    """
    PlotStyle.initialize()

    # Create figure with gridspec for precise layout control
    fig = plt.figure(figsize=CONFIG["PLOT"]["FIGURE_SIZE"])
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1, 1],
        wspace=0.2,
        left=0.1,
        right=0.9,
        top=0.9,
        bottom=0.15,
    )

    # Calculate common data
    n_prism = np.sqrt(float(structure.eps_prism))
    incident_angles = structure.incident_angle
    kx = n_prism * np.sin(incident_angles)
    frequency = structure.frequency
    max_kx = n_prism

    # Create first subplot
    ax1 = fig.add_subplot(gs[0])
    ax1.pcolormesh(kx, frequency, param1, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)

    # Create second subplot
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    im2 = ax2.pcolormesh(kx, frequency, param2, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)

    # Style both subplots
    [rotation_y1, rotation_y2]
    for idx, (ax, title, rot_y) in enumerate(
        [(ax1, title1, rotation_y1), (ax2, title2, rotation_y2)]
    ):
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
        positive_ticks = np.arange(0, max_tick + step / 2, step)
        negative_ticks = -np.arange(step, max_tick + step / 2, step)
        ticks = np.concatenate([negative_ticks, positive_ticks])
        ticks = ticks[np.abs(ticks) <= max_kx]
        ax.set_xticks(ticks)

        ax.set_ylim(frequency[0], frequency[-1])

        # Apply tick styling
        ax.tick_params(
            labelsize=CONFIG["PLOT"]["TICK_SIZE"],
            width=CONFIG["PLOT"]["TICK_WIDTH"],
            length=CONFIG["PLOT"]["TICK_LENGTH"],
            direction="in",
            pad=5,
            top=False,
            right=False,
        )

        # Set labels
        ax.set_xlabel(r"$k_x/k_0$", fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
        if idx == 0:  # Only add ylabel to first subplot
            ax.set_ylabel(
                r"$\omega/2\pi c$ (cm$^{-1}$)",
                fontsize=CONFIG["PLOT"]["LABEL_SIZE"],
                labelpad=10,
            )
        else:
            ax.tick_params(labelleft=False)

        # Add subplot label
        ax.text(
            0.06,
            0.94,
            f'({["a", "b"][idx]})',
            transform=ax.transAxes,
            fontsize=CONFIG["PLOT"]["TEXT_SIZE"],
            va="top",
            ha="left",
        )

        # Add rotation angle if provided
        if rot_y is not None:
            ax.text(
                0.98,
                0.96,
                rf"$\varphi = {rot_y}^{{\circ}}$",
                transform=ax.transAxes,
                fontsize=CONFIG["PLOT"]["TEXT_SIZE"],
                ha="right",
                va="top",
            )

        if title:
            ax.text(
                0.5,
                1.02,
                title,
                transform=ax.transAxes,
                fontsize=CONFIG["PLOT"]["LABEL_SIZE"],
                ha="center",
            )

        ax.set_box_aspect(1)

    # Add colorbar to the right of the second subplot
    pos = ax2.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0 + 0.12, 0.01, pos.height * 0.8])

    cbar = fig.colorbar(im2, cax=cbar_ax, shrink=0.8, aspect=2, ticks=[0, 0.5, 1])
    cbar.set_label("Reflectance", size=16)
    cbar.ax.yaxis.set_tick_params(
        labelsize=14, width=0, length=0, direction="in", right=True, left=True, top=True
    )

    # Save plot if filename provided
    if save_name:
        CONFIG["PLOT"]["SAVE_PATH"].mkdir(exist_ok=True)
        plt.savefig(
            CONFIG["PLOT"]["SAVE_PATH"] / f"{save_name}.png",
            dpi=CONFIG["PLOT"]["DPI"],
            bbox_inches="tight",
            pad_inches=0.1,
        )

    plt.show()
    plt.close()


def plot_mueller_dispersion(
    structure: Structure,
    param: np.ndarray,
    title: str | None = None,
    rotation_y: float | None = None,
    save_name: str | None = None,
    label: str = "a",
) -> None:
    """Plot k-space dispersion in kx-ky coordinates at fixed frequency.

    Args:
        structure: Structure object with incident_angle, azimuthal_angle arrays
        param: 2D parameter array [180, 480]
        title: Optional plot title
        rotation_y: Optional rotation angle to display
        save_name: Optional filename for saving
        label: Subplot label

    Note:
        Shows isofrequency contours in momentum space. The unit circle
        indicates the light cone (k = k0). Features outside indicate
        evanescent modes.
    """
    PlotStyle.initialize()

    # Create figure with gridspec for precise layout control
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(nrows=1, ncols=1, wspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.15)

    # Create axis with gridspec
    ax = fig.add_subplot(gs[0])

    # Calculate k-space coordinates
    incident_angle = structure.incident_angle
    z_rotation = structure.azimuthal_angle
    max_k = np.sqrt(float(structure.eps_prism))  # Maximum k value from prism

    # Create meshgrid for incident angle and z-rotation
    incident_angle, z_rotation = np.meshgrid(incident_angle, z_rotation)

    # Convert polar coordinates to Cartesian (kx, ky)
    kx = max_k * np.sin(incident_angle) * np.cos(z_rotation)
    ky = max_k * np.sin(incident_angle) * np.sin(z_rotation)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The input coordinates to pcolormesh")
        im = ax.pcolormesh(kx, ky, param.T, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)

    # Set plot limits and aspect ratio
    ax.set_aspect("equal")
    ax.set_xlim(-max_k * 1.1, max_k * 1.1)
    ax.set_ylim(-max_k * 1.1, max_k * 1.1)

    # Determine step size based on the range (same logic as plot_kx_frequency)
    if max_k < 3:
        step = 1  # Integer steps for small ranges
    elif max_k < 8:
        step = 2  # Steps of 2 for medium ranges
    elif max_k < 15:
        step = 3  # Steps of 3 for larger ranges
    else:
        step = 5  # Steps of 5 for very large ranges

    # Calculate maximum tick value
    max_tick = (int(max_k) // step) * step

    # Generate symmetrical ticks around zero
    positive_ticks = np.arange(0, max_tick + step / 2, step)
    negative_ticks = -np.arange(step, max_tick + step / 2, step)
    ticks = np.concatenate([negative_ticks, positive_ticks])
    ticks = ticks[np.abs(ticks) <= max_k]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # Apply paper-quality tick styling
    ax.tick_params(
        labelsize=CONFIG["PLOT"]["TICK_SIZE"],
        width=CONFIG["PLOT"]["TICK_WIDTH"],
        length=CONFIG["PLOT"]["TICK_LENGTH"],
        direction="in",
        pad=5,
        top=False,
        right=False,
    )

    # Set axis labels with LaTeX formatting
    ax.set_xlabel(r"$k_x/k_0$", fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
    ax.set_ylabel(r"$k_y/k_0$", fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)

    # Add subplot label in top left corner
    ax.text(
        0.06,
        0.94,
        f"({label})",
        transform=ax.transAxes,
        fontsize=CONFIG["PLOT"]["TEXT_SIZE"],
        va="top",
        ha="left",
    )

    # Add rotation angle if provided
    if rotation_y is not None:
        ax.text(
            0.98,
            0.96,
            rf"$\varphi = {rotation_y}^{{\circ}}$",
            transform=ax.transAxes,
            fontsize=CONFIG["PLOT"]["TEXT_SIZE"],
            ha="right",
            va="top",
        )

    # Add title if provided
    if title:
        ax.text(
            0.5,
            1.02,
            title,
            transform=ax.transAxes,
            fontsize=CONFIG["PLOT"]["LABEL_SIZE"],
            ha="center",
        )

    # Add unit circle to indicate light cone
    circle = plt.Circle((0, 0), 1, fill=False, color="white", linestyle="-", linewidth=1.5)
    ax.add_patch(circle)

    # Set aspect ratio to make plot square
    ax.set_box_aspect(1)

    # Add custom positioned colorbar
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0 + 0.12, 0.01, pos.height * 0.8])

    cbar = fig.colorbar(im, cax=cbar_ax, shrink=0.8, aspect=2, ticks=[0, 0.5, 1])
    cbar.set_label("Reflectance", size=16)
    cbar.ax.yaxis.set_tick_params(
        labelsize=14, width=0, length=0, direction="in", right=True, left=True, top=True
    )

    # Save plot if filename provided
    if save_name:
        CONFIG["PLOT"]["SAVE_PATH"].mkdir(exist_ok=True)
        plt.savefig(
            CONFIG["PLOT"]["SAVE_PATH"] / f"{save_name}.png",
            dpi=CONFIG["PLOT"]["DPI"],
            bbox_inches="tight",
            pad_inches=0.1,
        )

    plt.show()
    plt.close()


def plot_mueller_dispersion_pair(
    structure: Structure,
    param1: np.ndarray,
    param2: np.ndarray,
    rotation_y1: float | None = None,
    rotation_y2: float | None = None,
    title1: str | None = None,
    title2: str | None = None,
    save_name: str | None = None,
) -> None:
    """Plot two k-space dispersion diagrams side-by-side.

    Args:
        structure: Structure object with dispersion data
        param1: First parameter array [180, 480]
        param2: Second parameter array [180, 480]
        rotation_y1: Rotation angle for first plot
        rotation_y2: Rotation angle for second plot
        title1: Title for left panel
        title2: Title for right panel
        save_name: Optional filename for saving

    Note:
        Both plots share colorbar and have unit circles indicating
        light cone boundaries.
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
        bottom=0.15,
    )

    # Calculate common k-space coordinates
    incident_angle = structure.incident_angle
    z_rotation = structure.azimuthal_angle
    max_k = np.sqrt(float(structure.eps_prism))

    incident_angle, z_rotation = np.meshgrid(incident_angle, z_rotation)
    kx = max_k * np.sin(incident_angle) * np.cos(z_rotation)
    ky = max_k * np.sin(incident_angle) * np.sin(z_rotation)

    # Create first subplot
    ax1 = fig.add_subplot(gs[0])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The input coordinates to pcolormesh")
        ax1.pcolormesh(kx, ky, param1.T, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)

    # Create second subplot
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The input coordinates to pcolormesh")
        im2 = ax2.pcolormesh(kx, ky, param2.T, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)

    # Style both subplots
    for idx, (ax, title, rot_y) in enumerate(
        [(ax1, title1, rotation_y1), (ax2, title2, rotation_y2)]
    ):
        # Set plot limits and aspect ratio
        ax.set_aspect("equal")
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
            labelsize=CONFIG["PLOT"]["TICK_SIZE"],
            width=CONFIG["PLOT"]["TICK_WIDTH"],
            length=CONFIG["PLOT"]["TICK_LENGTH"],
            direction="in",
            pad=5,
            top=False,
            right=False,
        )

        # Set labels
        ax.set_xlabel(r"$k_x/k_0$", fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
        if idx == 0:  # Only add ylabel to first subplot
            ax.set_ylabel(r"$k_y/k_0$", fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
        else:
            ax.tick_params(labelleft=False)

        # Add subplot label
        ax.text(
            0.06,
            0.94,
            f'({["a", "b"][idx]})',
            transform=ax.transAxes,
            fontsize=CONFIG["PLOT"]["TEXT_SIZE"],
            va="top",
            ha="left",
        )

        # Add rotation angle if provided
        if rot_y is not None:
            ax.text(
                0.98,
                0.96,
                rf"$\varphi = {rot_y}^{{\circ}}$",
                transform=ax.transAxes,
                fontsize=CONFIG["PLOT"]["TEXT_SIZE"],
                ha="right",
                va="top",
            )

        if title:
            ax.text(
                0.5,
                1.02,
                title,
                transform=ax.transAxes,
                fontsize=CONFIG["PLOT"]["LABEL_SIZE"],
                ha="center",
            )

        # Add unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color="white", linestyle="-", linewidth=1.5)
        ax.add_patch(circle)

        ax.set_box_aspect(1)

    # Add colorbar to the right of the second subplot
    pos = ax2.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0 + 0.12, 0.01, pos.height * 0.8])

    cbar = fig.colorbar(im2, cax=cbar_ax, shrink=0.8, aspect=2, ticks=[0, 0.5, 1])
    cbar.set_label("Reflectance", size=16)
    cbar.ax.yaxis.set_tick_params(
        labelsize=14, width=0, length=0, direction="in", right=True, left=True, top=True
    )

    # Save plot if filename provided
    if save_name:
        CONFIG["PLOT"]["SAVE_PATH"].mkdir(exist_ok=True)
        plt.savefig(
            CONFIG["PLOT"]["SAVE_PATH"] / f"{save_name}.png",
            dpi=CONFIG["PLOT"]["DPI"],
            bbox_inches="tight",
            pad_inches=0.1,
        )

    plt.show()
    plt.close()
