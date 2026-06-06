"""Jones calculus for coherent (amplitude-level) polarization analysis.

Where :class:`~hyperbolic_optics.mueller.Mueller` works with Stokes vectors and
4×4 real Mueller matrices (intensity level, discards absolute phase and can model
depolarization), this module works with the 2×2 complex **Jones** matrix and
2-component Jones vectors — the amplitude-level description that *preserves phase*
and is the natural tool for fully-polarized, coherent light.

A structure's reflection coefficients are already a Jones matrix,

    J = [[r_pp, r_ps],
         [r_sp, r_ss]]   (rows = output p/s, columns = input p/s),

so :class:`Jones` builds it directly, applies ideal 2×2 optical components
(polarizer, wave plates, rotator) and an incident Jones vector, and exposes the
output state plus its Stokes parameters / polarization ellipse. ``to_mueller``
bridges to the Mueller formalism (reusing :meth:`Mueller._mueller_from_jones`) so
the two stay consistent.

Note on cascading: ideal components (polarizers, wave plates) are
angle-independent, so they compose with a sample by simple matrix multiplication
and broadcast freely over any scenario sweep. Cascading *two angle-dependent
structures* is only physical when they share the conserved in-plane wavevector
``kx`` (and frequency) — composing independent incident-angle sweeps (especially
with different prisms) is not meaningful, since ``kx`` differs point-to-point.
"""

from typing import Any

import numpy as np

from hyperbolic_optics.mueller import Mueller
from hyperbolic_optics.structure import Structure


class Jones:
    """Jones-calculus analyzer for coherent polarization (amplitude level).

    Mirrors :class:`~hyperbolic_optics.mueller.Mueller` but with 2×2 complex Jones
    matrices and 2-vectors, preserving phase. Batched over the scenario's
    presentation shape (the 2×2 component matrices broadcast against the sample's
    ``[..., 2, 2]`` Jones matrix, exactly as Mueller's 4×4 components do).

    Examples:
        >>> structure = Structure()
        >>> structure.execute(payload)
        >>> jones = Jones(structure)
        >>> jones.set_incident_polarization("linear", angle=0)   # p-polarized
        >>> jones.add_optical_component("sample")
        >>> jones.add_optical_component("linear_polarizer", 90)   # crossed analyzer
        >>> intensity = jones.get_intensity()
    """

    def __init__(self, structure: Structure) -> None:
        """Initialize from an executed structure.

        Args:
            structure: A :class:`Structure` on which ``execute`` has run.
        """
        self.structure = structure
        self.jones_matrix = None
        self.incident_jones = np.array([1.0, 0.0], dtype=np.complex128)  # default p-pol
        self.optical_components: list[np.ndarray] = []
        self.jones_vector = None

    # -- incident states ------------------------------------------------------

    def set_incident_polarization(self, polarization_type: str, **kwargs: Any) -> None:
        """Set the incident Jones vector ``[E_p, E_s]``.

        Args:
            polarization_type: ``"linear"`` (``angle`` deg, 0 = p, 90 = s),
                ``"circular"`` (``handedness`` ``"right"``/``"left"``), or
                ``"elliptical"`` (``alpha`` deg azimuth, ``ellipticity`` deg).

        Raises:
            ValueError: If ``polarization_type`` is unknown.
        """
        if polarization_type == "linear":
            angle = np.radians(kwargs.get("angle", 0.0))
            self.incident_jones = np.array([np.cos(angle), np.sin(angle)], dtype=np.complex128)
        elif polarization_type == "circular":
            # right -> [1, i]/sqrt2 gives Stokes S3 = +1 (matches Mueller's convention)
            sign = 1.0j if kwargs.get("handedness", "right") == "right" else -1.0j
            self.incident_jones = np.array([1.0, sign], dtype=np.complex128) / np.sqrt(2.0)
        elif polarization_type == "elliptical":
            alpha = np.radians(kwargs.get("alpha", 0.0))
            chi = np.radians(kwargs.get("ellipticity", 0.0))
            base = np.array([np.cos(chi), 1.0j * np.sin(chi)], dtype=np.complex128)
            self.incident_jones = self.rotator(np.degrees(alpha)) @ base
        else:
            raise ValueError(f"Unsupported polarization type: {polarization_type}")

    # -- 2x2 component matrices ----------------------------------------------

    @staticmethod
    def rotator(angle: float) -> np.ndarray:
        """Rotation matrix by ``angle`` degrees in the (p, s) plane."""
        a = np.radians(angle)
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.complex128)

    def linear_polarizer(self, angle: float) -> np.ndarray:
        """Ideal linear polarizer with transmission axis at ``angle`` degrees."""
        a = np.radians(angle)
        cos, sin = np.cos(a), np.sin(a)
        return np.array([[cos**2, cos * sin], [cos * sin, sin**2]], dtype=np.complex128)

    def quarter_wave_plate(self, angle: float) -> np.ndarray:
        """Quarter-wave plate with fast axis at ``angle`` degrees (π/2 retardance)."""
        retarder = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
        return self.rotator(angle) @ retarder @ self.rotator(-angle)

    def half_wave_plate(self, angle: float) -> np.ndarray:
        """Half-wave plate with fast axis at ``angle`` degrees (π retardance)."""
        retarder = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        return self.rotator(angle) @ retarder @ self.rotator(-angle)

    # -- sample Jones matrix --------------------------------------------------

    def calculate_jones_matrix(self, transmission: bool = False) -> np.ndarray:
        """Build the sample's 2×2 Jones matrix from its r (or t) coefficients.

        Args:
            transmission: If True, use the transmission coefficients ``t_*``
                (computing them via ``Structure.calculate_transmissivity`` if
                needed) instead of the reflection coefficients.

        Returns:
            The Jones matrix ``[..., 2, 2]`` in the scenario's presentation shape.
        """
        if transmission:
            if self.structure.t_pp is None:
                self.structure.calculate_transmissivity()
            pp, ps, sp, ss = (
                self.structure.t_pp,
                self.structure.t_ps,
                self.structure.t_sp,
                self.structure.t_ss,
            )
        else:
            pp, ps, sp, ss = (
                self.structure.r_pp,
                self.structure.r_ps,
                self.structure.r_sp,
                self.structure.r_ss,
            )
        jones = np.array([[pp, ps], [sp, ss]], dtype=np.complex128)
        # 2x2 built as leading axes -> move to trailing so it broadcasts for any
        # batch rank (scalar Simple, 2-D Incident/Azimuthal/Dispersion, 3-D FullSweep).
        self.jones_matrix = np.moveaxis(jones, (0, 1), (-2, -1))
        return self.jones_matrix

    def add_optical_component(self, component_type: str, *args: Any) -> None:
        """Append an optical element to the path (applied left-to-right on the beam).

        Args:
            component_type: ``"sample"`` (the structure's reflection Jones matrix),
                ``"sample_transmission"``, ``"linear_polarizer"``,
                ``"quarter_wave_plate"``, ``"half_wave_plate"`` or ``"rotator"``.
            *args: Angle (degrees) for the ideal components.

        Raises:
            ValueError: If ``component_type`` is unknown.
        """
        if component_type == "sample":
            self.optical_components.append(self.calculate_jones_matrix(transmission=False))
        elif component_type == "sample_transmission":
            self.optical_components.append(self.calculate_jones_matrix(transmission=True))
        elif component_type == "linear_polarizer":
            self.optical_components.append(self.linear_polarizer(*args))
        elif component_type == "quarter_wave_plate":
            self.optical_components.append(self.quarter_wave_plate(*args))
        elif component_type == "half_wave_plate":
            self.optical_components.append(self.half_wave_plate(*args))
        elif component_type == "rotator":
            self.optical_components.append(self.rotator(*args))
        else:
            raise ValueError(f"Unsupported optical component type: {component_type}")

    # -- output state and derived quantities ---------------------------------

    def calculate_jones_vector(self) -> np.ndarray:
        """Propagate the incident Jones vector through the component chain.

        Returns:
            Output Jones vector ``[..., 2]`` = ``[E_p, E_s]`` (presentation shape).
        """
        vector = self.incident_jones.reshape(2, 1)
        for component in self.optical_components:
            vector = component @ vector
        self.jones_vector = vector[..., 0]
        return self.jones_vector

    def _ensure_vector(self) -> np.ndarray:
        if self.jones_vector is None:
            self.calculate_jones_vector()
        return self.jones_vector

    def get_intensity(self) -> np.ndarray:
        """Output intensity ``|E_p|² + |E_s|²`` (presentation shape)."""
        vector = self._ensure_vector()
        return np.abs(vector[..., 0]) ** 2 + np.abs(vector[..., 1]) ** 2

    def get_stokes_parameters(self) -> dict[str, np.ndarray]:
        """Stokes parameters of the output Jones vector.

        Convention (``e^{-iωt}``, p↔x, s↔y): ``S0=|Ep|²+|Es|²``, ``S1=|Ep|²−|Es|²``,
        ``S2=2Re(Ep Es*)``, ``S3=−2Im(Ep Es*)`` — consistent with
        :meth:`FieldProfile.stokes_from_field_profile` and :class:`Mueller`.
        """
        e_p, e_s = self._ensure_vector()[..., 0], self._ensure_vector()[..., 1]
        return {
            "S0": np.abs(e_p) ** 2 + np.abs(e_s) ** 2,
            "S1": np.abs(e_p) ** 2 - np.abs(e_s) ** 2,
            "S2": 2.0 * np.real(e_p * np.conj(e_s)),
            "S3": -2.0 * np.imag(e_p * np.conj(e_s)),
        }

    def get_ellipse(self) -> dict[str, np.ndarray]:
        """Polarization ellipse of the output: ``azimuth`` ψ and ``ellipticity`` χ (rad)."""
        stokes = self.get_stokes_parameters()
        s1, s2, s3 = stokes["S1"], stokes["S2"], stokes["S3"]
        return {
            "azimuth": 0.5 * np.arctan2(s2, s1),
            "ellipticity": 0.5 * np.arctan2(s3, np.sqrt(s1**2 + s2**2)),
        }

    def eigenpolarizations(self, transmission: bool = False) -> dict[str, np.ndarray]:
        """Eigenpolarizations of the sample's Jones matrix (states with no conversion).

        An eigenvector ``v`` of the Jones matrix satisfies ``J·v = λ·v``: it reflects
        (or transmits) into the *same* polarization state, scaled by the complex
        eigenvalue ``λ`` (amplitude + phase). These are the polarization states that
        undergo **no polarization conversion**.

        Because a lossy/anisotropic reflection Jones matrix is generally
        **non-normal**, the two eigenpolarizations are not orthogonal and can
        **coalesce at an exceptional point (EP)**, where the matrix becomes
        defective (non-diagonalizable). Two EP diagnostics are returned: the
        ``discriminant`` ``D = ((J_pp − J_ss)/2)² + J_ps·J_sp`` (``|D| → 0`` at an EP,
        since ``λ± = tr/2 ± √D`` coalesce) and the ``eigenvector_overlap``
        ``|⟨v₀|v₁⟩|`` of the unit eigenvectors (``→ 1`` at an EP; ``0`` when orthogonal).

        Args:
            transmission: Use the transmission Jones matrix instead of reflection.

        Returns:
            Dict with ``eigenvalues`` ``[..., 2]``, ``eigenpolarizations``
            ``[..., 2, 2]`` (columns are the unit eigenvectors), ``discriminant``
            ``[...]``, and ``eigenvector_overlap`` ``[...]``.
        """
        jones = self.calculate_jones_matrix(transmission=transmission)
        eigenvalues, eigenvectors = np.linalg.eig(jones)
        j_pp, j_ps = jones[..., 0, 0], jones[..., 0, 1]
        j_sp, j_ss = jones[..., 1, 0], jones[..., 1, 1]
        discriminant = (0.5 * (j_pp - j_ss)) ** 2 + j_ps * j_sp
        v0, v1 = eigenvectors[..., :, 0], eigenvectors[..., :, 1]
        overlap = np.abs(np.sum(np.conj(v0) * v1, axis=-1))  # eig returns unit vectors
        return {
            "eigenvalues": eigenvalues,
            "eigenpolarizations": eigenvectors,
            "discriminant": discriminant,
            "eigenvector_overlap": overlap,
        }

    def to_mueller(self, transmission: bool = False) -> np.ndarray:
        """Mueller matrix of the sample, bridged from its Jones matrix.

        Reuses :meth:`Mueller._mueller_from_jones`, so the Jones and Mueller
        descriptions of the same structure are guaranteed consistent.
        """
        if transmission:
            if self.structure.t_pp is None:
                self.structure.calculate_transmissivity()
            pp, ps, sp, ss = (
                self.structure.t_pp,
                self.structure.t_ps,
                self.structure.t_sp,
                self.structure.t_ss,
            )
        else:
            pp, ps, sp, ss = (
                self.structure.r_pp,
                self.structure.r_ps,
                self.structure.r_sp,
                self.structure.r_ss,
            )
        return Mueller._mueller_from_jones(pp, ps, sp, ss)
