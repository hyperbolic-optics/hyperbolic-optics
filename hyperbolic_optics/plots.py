"""Publication-quality plotting functions for simulation results.

This module provides plotting utilities for various visualization types:

    - Material permittivity spectra
    - Frequency vs incident angle (kx-ω diagrams)
    - Frequency vs azimuthal angle (β-ω diagrams)
    - k-space dispersion (kx-ky isofrequency contours)
    - Stokes parameter distributions
    - Mueller matrix elements

All plots share a small set of styling helpers (see the ``_`` prefixed
functions) for consistent, publication-ready figures with proper axis labels,
colorbars, and typography.
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
        "FIGURE_SIZE": (20, 5),
    },
    "PHYSICS": {"PRISM_PERMITTIVITY": 50.0},
}

_FREQ_LABEL = r"$\omega/2\pi c$ (cm$^{-1}$)"


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


# --- shared figure/styling helpers -------------------------------------------


def _single_axis(figsize: tuple[float, float] = (10, 5)) -> tuple[plt.Figure, plt.Axes]:
    """Create a one-panel figure with the standard gridspec margins."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=1, ncols=1, wspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.15)
    return fig, fig.add_subplot(gs[0])


def _pair_axes(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Create a two-panel figure (shared y-axis) with the standard gridspec margins."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.15
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    return fig, ax1, ax2


def _style_heatmap_ticks(ax: plt.Axes) -> None:
    """Apply the paper-quality tick styling used by every heatmap panel."""
    ax.tick_params(
        labelsize=CONFIG["PLOT"]["TICK_SIZE"],
        width=CONFIG["PLOT"]["TICK_WIDTH"],
        length=CONFIG["PLOT"]["TICK_LENGTH"],
        direction="in",
        pad=5,
        top=False,
        right=False,
    )


def _symmetric_ticks(max_val: float, step: float) -> np.ndarray:
    """Symmetric ticks about zero at the given step, clipped to ``|t| <= max_val``."""
    max_tick = (int(max_val) // step) * step
    positive = np.arange(0, max_tick + step / 2, step)
    negative = -np.arange(step, max_tick + step / 2, step)
    ticks = np.concatenate([negative, positive])
    return ticks[np.abs(ticks) <= max_val]


def _subplot_label(ax: plt.Axes, label: str) -> None:
    """Add a ``(a)``-style subplot label in the top-left corner."""
    ax.text(
        0.06,
        0.94,
        f"({label})",
        transform=ax.transAxes,
        fontsize=CONFIG["PLOT"]["TEXT_SIZE"],
        va="top",
        ha="left",
    )


def _panel_title(ax: plt.Axes, title: str | None) -> None:
    """Add a centered title above the panel, if provided."""
    if title:
        ax.text(
            0.5,
            1.02,
            title,
            transform=ax.transAxes,
            fontsize=CONFIG["PLOT"]["LABEL_SIZE"],
            ha="center",
        )


def _rotation_text(ax: plt.Axes, rotation_y: float | None) -> None:
    """Annotate the panel with ``φ = <rotation_y>°`` in the top-right, if provided."""
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


def _reflectance_colorbar(fig: plt.Figure, ax: plt.Axes, im) -> None:
    """Add the custom-positioned 'Reflectance' colorbar to the right of ``ax``."""
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0 + 0.12, 0.01, pos.height * 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax, shrink=0.8, aspect=2, ticks=[0, 0.5, 1])
    cbar.set_label("Reflectance", size=16)
    cbar.ax.yaxis.set_tick_params(
        labelsize=14, width=0, length=0, direction="in", right=True, left=True, top=True
    )


def _save_and_show(save_name: str | None) -> None:
    """Save the current figure (if ``save_name`` given), then show and close it."""
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


def _kx_axis(structure: Structure) -> tuple[np.ndarray, float]:
    """Return (kx array, max_kx) for a dispersion/incident structure."""
    n_prism = np.sqrt(float(structure.eps_prism))
    return n_prism * np.sin(structure.incident_angle), n_prism


def _kspace_grid(structure: Structure) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (kx, ky, max_k) meshgrids for a k-space dispersion plot."""
    max_k = np.sqrt(float(structure.eps_prism))
    incident_angle, z_rotation = np.meshgrid(structure.incident_angle, structure.azimuthal_angle)
    kx = max_k * np.sin(incident_angle) * np.cos(z_rotation)
    ky = max_k * np.sin(incident_angle) * np.sin(z_rotation)
    return kx, ky, max_k


def _light_cone(ax: plt.Axes) -> None:
    """Draw the unit circle (light cone, k = k0) on a k-space panel."""
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="white", linestyle="-", linewidth=1.5))


# --- public plotting functions -----------------------------------------------


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

    axs[0].plot(material.frequency, eps_ext.real, label=r"$\mathrm{Re}(\varepsilon_\mathrm{ext})$")
    axs[0].plot(material.frequency, eps_ord.real, label=r"$\mathrm{Re}(\varepsilon_\mathrm{ord})$")
    axs[0].axhline(y=0, color="black", linewidth=1)
    axs[0].set(ylabel=r"$\mathrm{Re}(\epsilon)$")
    axs[0].legend()
    PlotStyle.style_axis(axs[0])

    axs[1].plot(material.frequency, eps_ext.imag, label=r"$\mathrm{Im}(\varepsilon_\mathrm{ext})$")
    axs[1].plot(material.frequency, eps_ord.imag, label=r"$\mathrm{Im}(\varepsilon_\mathrm{ord})$")
    axs[1].set(xlabel=r"Wavenumber (cm$^{-1}$)", ylabel=r"$\mathrm{Im}(\epsilon)$")
    axs[1].set_xlim(material.frequency[0], material.frequency[-1])
    axs[1].set_ylim(0)
    axs[1].legend()
    PlotStyle.style_axis(axs[1])

    _save_and_show(save_name)


def plot_permittivity_tensor(material: BaseMaterial, save_name: str | None = None) -> None:
    """Plot Re/Im of the diagonal permittivity components for any material.

    General over crystal class: uniaxial (xx == yy, distinct zz), biaxial (xx, yy,
    zz all distinct) and monoclinic (diagonal shown; off-diagonal coupling not
    drawn). Uses the material's own frequency grid.

    Args:
        material: A material with its ``frequency`` array set (e.g. any entry from
            :func:`hyperbolic_optics.materials.list_materials`).
        save_name: Optional filename for saving (without extension).

    Raises:
        ValueError: If the material has no intrinsic frequency range.
    """
    if material.frequency is None:
        raise ValueError("Material has no frequency range; set material.frequency first.")
    PlotStyle.initialize()

    frequency = material.frequency
    tensor = material.fetch_permittivity_tensor()  # [F, 3, 3]
    components = {
        r"$\varepsilon_{xx}$": tensor[:, 0, 0],
        r"$\varepsilon_{yy}$": tensor[:, 1, 1],
        r"$\varepsilon_{zz}$": tensor[:, 2, 2],
    }

    fig, axs = plt.subplots(2, figsize=(9, 7), sharex=True, gridspec_kw={"hspace": 0.1})
    for label, eps in components.items():
        axs[0].plot(frequency, eps.real, label=label)
        axs[1].plot(frequency, eps.imag, label=label)
    axs[0].axhline(y=0, color="black", linewidth=1)
    axs[0].set(ylabel=r"$\mathrm{Re}(\epsilon)$", title=getattr(material, "name", ""))
    axs[0].legend()
    PlotStyle.style_axis(axs[0])
    axs[1].set(xlabel=r"Wavenumber (cm$^{-1}$)", ylabel=r"$\mathrm{Im}(\epsilon)$")
    axs[1].set_xlim(frequency[0], frequency[-1])
    axs[1].legend()
    PlotStyle.style_axis(axs[1])

    _save_and_show(save_name)


def plot_materials_gallery(save_name: str | None = None) -> None:
    """Grid of Re(ε) vs frequency for every built-in material.

    A quick visual check of the phonon parameters across the catalogue
    (:func:`hyperbolic_optics.materials.list_materials`) — e.g. hBN's two bands,
    MoO₃'s three, and the hyperbolic sign flips. ε_xx, ε_yy, ε_zz are drawn over
    each material's default frequency range.

    Args:
        save_name: Optional filename for saving (without extension).
    """
    from hyperbolic_optics.materials import create_material, list_materials

    PlotStyle.initialize()
    names = list(list_materials())
    ncols = 3
    nrows = -(-len(names) // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        material = create_material(name)
        tensor = material.fetch_permittivity_tensor()
        for idx, comp in enumerate(("xx", "yy", "zz")):
            ax.plot(
                material.frequency, tensor[:, idx, idx].real, label=rf"$\varepsilon_{{{comp}}}$"
            )
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_title(name)
        ax.set(xlabel=r"$\omega$ (cm$^{-1}$)", ylabel=r"$\mathrm{Re}(\epsilon)$")
        ax.legend(fontsize=8, ncol=3)
    for ax in axes[len(names) :]:
        ax.axis("off")

    fig.tight_layout()
    _save_and_show(save_name)


def plot_poincare_sphere(
    stokes: dict[str, np.ndarray] | np.ndarray,
    color_by: np.ndarray | None = None,
    title: str | None = None,
    save_name: str | None = None,
) -> None:
    """Plot a Stokes trajectory on the Poincaré sphere (works for any scenario).

    Takes the output Stokes parameters from any scenario (e.g.
    ``Mueller.get_stokes_parameters`` or ``Jones.get_stokes_parameters``) and plots
    the normalized state ``(S1, S2, S3)/S0`` on the unit sphere — poles are right/
    left circular, the equator is linear. All batch axes are flattened into the
    point cloud, so it visualizes how polarization evolves across an incident-angle,
    azimuthal, frequency or depth sweep.

    Args:
        stokes: Either a dict with keys ``S0, S1, S2, S3`` or an array ``[..., 4]``.
        color_by: Optional array (any broadcastable shape) used to colour the
            points — e.g. frequency or sweep index. Defaults to point order.
        title: Optional plot title.
        save_name: Optional filename for saving (without extension).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    PlotStyle.initialize()
    if isinstance(stokes, dict):
        s0, s1, s2, s3 = (np.asarray(stokes[k], dtype=np.float64) for k in ("S0", "S1", "S2", "S3"))
    else:
        arr = np.asarray(stokes, dtype=np.float64)
        s0, s1, s2, s3 = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]

    s0_safe = np.where(np.abs(s0) > 1e-12, s0, 1.0)
    x = (s1 / s0_safe).ravel()
    y = (s2 / s0_safe).ravel()
    z = (s3 / s0_safe).ravel()
    colours = np.arange(x.size) if color_by is None else np.broadcast_to(color_by, s0.shape).ravel()

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    ax.plot_wireframe(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        color="0.85",
        linewidth=0.4,
    )
    scatter = ax.scatter(x, y, z, c=colours, cmap="viridis", s=8)
    fig.colorbar(
        scatter, ax=ax, shrink=0.6, pad=0.1, label="frequency" if color_by is not None else "order"
    )
    ax.set_xlabel(r"$S_1/S_0$")
    ax.set_ylabel(r"$S_2/S_0$")
    ax.set_zlabel(r"$S_3/S_0$")
    ax.set_box_aspect((1, 1, 1))
    if title:
        ax.set_title(title)
    _save_and_show(save_name)


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
        Color plot with frequency on the y-axis and azimuthal angle (β) on the
        x-axis, for studying rotational anisotropy.
    """
    PlotStyle.initialize()
    fig, ax = _single_axis()

    x_axis = np.round(np.degrees(structure.azimuthal_angle), 1)
    frequency = structure.frequency
    im = ax.pcolormesh(x_axis, frequency, param, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)

    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_ylim(frequency[0], frequency[-1])
    _style_heatmap_ticks(ax)
    ax.set_xlabel(r"$\beta$ (degree)", fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
    ax.set_ylabel(_FREQ_LABEL, fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
    _subplot_label(ax, label)
    _panel_title(ax, title)
    ax.set_box_aspect(1)

    _reflectance_colorbar(fig, ax, im)
    _save_and_show(save_name)


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
    fig, ax1, ax2 = _pair_axes(CONFIG["PLOT"]["FIGURE_SIZE"])

    x_axis = np.round(np.degrees(structure.azimuthal_angle), 1)
    frequency = structure.frequency
    ax1.pcolormesh(x_axis, frequency, param1, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)
    im2 = ax2.pcolormesh(x_axis, frequency, param2, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)

    for idx, (ax, title) in enumerate([(ax1, title1), (ax2, title2)]):
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_ylim(frequency[0], frequency[-1])
        _style_heatmap_ticks(ax)
        ax.set_xlabel(r"$\beta$ (degree)", fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
        if idx == 0:
            ax.set_ylabel(_FREQ_LABEL, fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
        else:
            ax.tick_params(labelleft=False)
        _subplot_label(ax, ["a", "b"][idx])
        _panel_title(ax, title)
        ax.set_box_aspect(1)

    _reflectance_colorbar(fig, ax2, im2)
    _save_and_show(save_name)


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
        ax[row, col].set_ylabel(_FREQ_LABEL)
        PlotStyle.style_axis(ax[row, col])

    plt.tight_layout()
    _save_and_show(save_name)


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
    fig, ax = _single_axis()

    kx, max_kx = _kx_axis(structure)
    frequency = structure.frequency
    im = ax.pcolormesh(kx, frequency, param, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)

    ax.set_xlim(-max_kx, max_kx)
    if max_kx < 3:
        step = 1
    elif max_kx < 8:
        step = 2
    elif max_kx < 15:
        step = 3
    else:
        step = 5
    ax.set_xticks(_symmetric_ticks(max_kx, step))
    ax.set_ylim(frequency[0], frequency[-1])
    _style_heatmap_ticks(ax)
    ax.set_xlabel(r"$k_x/k_0$", fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
    ax.set_ylabel(_FREQ_LABEL, fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
    _subplot_label(ax, label)
    _rotation_text(ax, rotation_y)
    _panel_title(ax, title)
    ax.set_box_aspect(1)

    _reflectance_colorbar(fig, ax, im)
    _save_and_show(save_name)


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
    fig, ax1, ax2 = _pair_axes(CONFIG["PLOT"]["FIGURE_SIZE"])

    kx, max_kx = _kx_axis(structure)
    frequency = structure.frequency
    ax1.pcolormesh(kx, frequency, param1, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)
    im2 = ax2.pcolormesh(kx, frequency, param2, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)

    for idx, (ax, title, rot_y) in enumerate(
        [(ax1, title1, rotation_y1), (ax2, title2, rotation_y2)]
    ):
        ax.set_xlim(-max_kx, max_kx)
        if max_kx < 3:
            step = 0.5
        elif max_kx < 8:
            step = 1
        elif max_kx < 15:
            step = 3
        else:
            step = 5
        ax.set_xticks(_symmetric_ticks(max_kx, step))
        ax.set_ylim(frequency[0], frequency[-1])
        _style_heatmap_ticks(ax)
        ax.set_xlabel(r"$k_x/k_0$", fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
        if idx == 0:
            ax.set_ylabel(_FREQ_LABEL, fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
        else:
            ax.tick_params(labelleft=False)
        _subplot_label(ax, ["a", "b"][idx])
        _rotation_text(ax, rot_y)
        _panel_title(ax, title)
        ax.set_box_aspect(1)

    _reflectance_colorbar(fig, ax2, im2)
    _save_and_show(save_name)


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
        indicates the light cone (k = k0); features outside are evanescent.
    """
    PlotStyle.initialize()
    fig, ax = _single_axis()

    kx, ky, max_k = _kspace_grid(structure)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The input coordinates to pcolormesh")
        im = ax.pcolormesh(kx, ky, param.T, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)

    ax.set_aspect("equal")
    ax.set_xlim(-max_k * 1.1, max_k * 1.1)
    ax.set_ylim(-max_k * 1.1, max_k * 1.1)
    if max_k < 3:
        step = 1
    elif max_k < 8:
        step = 2
    elif max_k < 15:
        step = 3
    else:
        step = 5
    ticks = _symmetric_ticks(max_k, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    _style_heatmap_ticks(ax)
    ax.set_xlabel(r"$k_x/k_0$", fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
    ax.set_ylabel(r"$k_y/k_0$", fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
    _subplot_label(ax, label)
    _rotation_text(ax, rotation_y)
    _panel_title(ax, title)
    _light_cone(ax)
    ax.set_box_aspect(1)

    _reflectance_colorbar(fig, ax, im)
    _save_and_show(save_name)


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
        Both plots share a colorbar and have unit circles indicating the
        light-cone boundary.
    """
    PlotStyle.initialize()
    fig, ax1, ax2 = _pair_axes((10, 5))

    kx, ky, max_k = _kspace_grid(structure)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The input coordinates to pcolormesh")
        ax1.pcolormesh(kx, ky, param1.T, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)
        im2 = ax2.pcolormesh(kx, ky, param2.T, cmap=CONFIG["PLOT"]["COLORMAP"], vmin=0, vmax=1)

    for idx, (ax, title, rot_y) in enumerate(
        [(ax1, title1, rotation_y1), (ax2, title2, rotation_y2)]
    ):
        ax.set_aspect("equal")
        ax.set_xlim(-max_k * 1.05, max_k * 1.05)
        ax.set_ylim(-max_k * 1.05, max_k * 1.05)

        if max_k < 3:
            tick_spacing = 1
        elif max_k < 6:
            tick_spacing = 2
        else:
            tick_spacing = 3
        neg_ticks = np.arange(0, -int(max_k) - 1, -tick_spacing)
        pos_ticks = np.arange(0, int(max_k) + 1, tick_spacing)
        ticks = np.concatenate([neg_ticks[1:], pos_ticks])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        _style_heatmap_ticks(ax)
        ax.set_xlabel(r"$k_x/k_0$", fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
        if idx == 0:
            ax.set_ylabel(r"$k_y/k_0$", fontsize=CONFIG["PLOT"]["LABEL_SIZE"], labelpad=10)
        else:
            ax.tick_params(labelleft=False)
        _subplot_label(ax, ["a", "b"][idx])
        _rotation_text(ax, rot_y)
        _panel_title(ax, title)
        _light_cone(ax)
        ax.set_box_aspect(1)

    _reflectance_colorbar(fig, ax2, im2)
    _save_and_show(save_name)
