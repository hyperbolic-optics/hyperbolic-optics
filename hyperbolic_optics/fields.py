"""Numerical transmission, layer-resolved absorption, and field profiles.

Where :mod:`hyperbolic_optics.structure` stops at reflection coefficients, this
module reconstructs the field through the stack and derives *power* quantities
from it directly (no closed-form transmission formulas). The tangential field
``F = [Ex, Ey, Hx, Hy]`` is continuous across interfaces and propagated by the
stored layer matrices (``Gᵢ = Mᵢ · G_{i+1}``); the normal energy flux is
``S_z = ½ Re(Ex·Hy* − Ey·Hx*)``. For unit incident power this gives reflectance
``R = 1 − S_z(G₁)/S_z^inc``, transmittance ``T = S_z(G_exit)/S_z^inc``, and
layer absorption ``Aᵢ = [S_z(Gᵢ) − S_z(G_{i+1})]/S_z^inc``, which conserve energy
exactly (``R + T + ΣAᵢ = 1``). For a single semi-infinite anisotropic exit there
are no interior layers and :meth:`FieldProfile.field_profile` shows ``T`` absorbed
with depth (decay length set by ``Im(kz)``).

References:

- Passler, Jeannin & Paarmann, JOSA B 37, 1060 (2020); arXiv:2002.03832
- pyGTM: ``System.calculate_Efield`` / ``calculate_Poynting_Absorption_vs_z``
- Passler & Paarmann, JOSA B 34, 2128-2139 (2017)
"""

from typing import Any

import numpy as np

from hyperbolic_optics.axes import T, present
from hyperbolic_optics.structure import Structure

# Amplitude-vector slot ordering shared by the prism and exit dynamical matrices,
# verified against AmbientIncidentMedium/AmbientExitMedium: [s_fwd, s_bwd, p_fwd,
# p_bwd]. Forward (= transmitted/incident) waves live in slots 0 and 2.
_S_FWD, _S_BWD, _P_FWD, _P_BWD = 0, 1, 2, 3


def _matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Batched matrix-vector product ``matrix @ vector`` over trailing 4-axes.

    ``matrix`` is ``[..., 4, 4]`` and ``vector`` is ``[..., 4]``; the leading batch
    axes broadcast (e.g. an ``[A, 1, 1, 4, 4]`` prism against an ``[A, B, F, 4]``
    field). Returns ``[..., 4]``.
    """
    return np.squeeze(matrix @ vector[..., np.newaxis], axis=-1)


def _poynting_z(field: np.ndarray) -> np.ndarray:
    """Time-averaged ``S_z = ½ Re(Ex·Hy* − Ey·Hx*)`` from ``F = [Ex, Ey, Hx, Hy]``.

    Accepts any trailing-4 array ``[..., 4]`` and returns ``[...]``.
    """
    Ex, Ey, Hx, Hy = field[..., 0], field[..., 1], field[..., 2], field[..., 3]
    return 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))


class FieldProfile:
    """Numerical transmission, layer-resolved absorption, and field profiles.

    Consumes an executed :class:`~hyperbolic_optics.structure.Structure` (mirrors
    how :class:`~hyperbolic_optics.mueller.Mueller` consumes one) and exposes the
    power quantities and field reconstruction described in the module docstring.

    The engine is fully batched over the canonical ``[A, B, F]`` axes, so
    :meth:`transmittance`, :meth:`reflectance` and :meth:`layer_absorption` return
    arrays in the same presentation layout as ``structure.r_pp``. :meth:`field_profile`
    adds a depth axis; it broadcasts too, but a full angle/frequency sweep ×
    ``n_points`` × 6 field components is multi-gigabyte, so it is intended for the
    ``Simple`` scenario (or a single ``(angle, frequency)`` slice).

    Examples:
        >>> structure = Structure()
        >>> structure.execute(payload)
        >>> fp = FieldProfile(structure)
        >>> fp.transmittance("p")          # power transmittance T
        >>> fp.layer_absorption("p")       # per-interior-layer absorption
        >>> fp.check_conservation("p")     # max |R + T + ΣA − 1|
        >>> prof = fp.field_profile("p")   # dict: z, Ex..Hz, Sz, absorption_*
    """

    def __init__(self, structure: Structure) -> None:
        """Cache the executed structure's transfer matrices and propagation constants.

        Args:
            structure: A :class:`Structure` on which ``execute`` has already run.

        Raises:
            ValueError: If the structure has not been executed (no transfer matrix).
        """
        if structure.transfer_matrix is None:
            raise ValueError("Structure has not been executed; call structure.execute(payload).")
        self.structure = structure
        self.layers = structure.layers
        self.n_layers = len(self.layers)
        self.gamma = structure.transfer_matrix  # canonical [A, B, F, 4, 4]
        self.k_x = np.asarray(structure.k_x, dtype=np.complex128)  # [A, 1, 1]
        self.k_0 = np.asarray(structure.k_0, dtype=np.complex128)  # [1, 1, F]
        # Incident dynamical matrix A_inc = inv(prism transfer matrix); maps mode
        # amplitudes -> tangential field in the prism. [A, 1, 1, 4, 4].
        self.a_inc = np.linalg.inv(self.layers[0].matrix)

    # -- amplitudes -----------------------------------------------------------

    @staticmethod
    def _resolve_polarization(
        polarization: str | tuple[complex, complex],
    ) -> tuple[complex, complex]:
        """Return the incident ``(a_s, a_p)`` forward amplitudes for a polarization.

        Accepts ``"p"``, ``"s"``, or an explicit complex ``(a_s, a_p)`` pair (a
        general Jones state; everything downstream is linear in these amplitudes).
        """
        if isinstance(polarization, str):
            key = polarization.lower()
            if key == "p":
                return 0.0 + 0j, 1.0 + 0j
            if key == "s":
                return 1.0 + 0j, 0.0 + 0j
            raise ValueError(f"Unknown polarization {polarization!r}; use 'p', 's', or (a_s, a_p).")
        a_s, a_p = polarization
        return complex(a_s), complex(a_p)

    def _transmitted_amplitudes(self, a_s: complex, a_p: complex) -> np.ndarray:
        """Solve for the exit-medium amplitude vector ``c_exit`` (transmitted only).

        Transmitted waves occupy forward slots ``(s_fwd, p_fwd)`` of the exit
        amplitude vector; backward slots are zero. Imposing that, ``c_inc = Γ·c_exit``
        reduces to a 2×2 system on rows/cols ``(0, 2)`` whose determinant is exactly
        the ``bottom_line`` of the analytic reflection formulas -- so this matches
        the existing coefficients while staying batched and free of closed forms.

        Returns:
            ``c_exit`` with shape ``[A, B, F, 4]`` (nonzero only in slots 0 and 2).
        """
        g = self.gamma
        # 2x2 sub-system [[g00, g02], [g20, g22]] @ [c_s, c_p] = [a_s, a_p].
        row0 = np.stack([g[..., _S_FWD, _S_FWD], g[..., _S_FWD, _P_FWD]], axis=-1)
        row1 = np.stack([g[..., _P_FWD, _S_FWD], g[..., _P_FWD, _P_FWD]], axis=-1)
        m2 = np.stack([row0, row1], axis=-2)  # [A, B, F, 2, 2]

        rhs = np.array([a_s, a_p], dtype=np.complex128)
        rhs = np.broadcast_to(rhs, m2.shape[:-2] + (2,))[..., np.newaxis]
        solved = np.linalg.solve(m2, rhs)[..., 0]  # [A, B, F, 2] = (c_s_fwd, c_p_fwd)

        c_exit = np.zeros(self.gamma.shape[:-1], dtype=np.complex128)  # [A, B, F, 4]
        c_exit[..., _S_FWD] = solved[..., 0]
        c_exit[..., _P_FWD] = solved[..., 1]
        return c_exit

    def _incident_flux(self, a_s: complex, a_p: complex) -> np.ndarray:
        """Reference flux ``S_z^inc`` of the incident wave alone, in the prism.

        Built from the forward columns of ``A_inc`` (the incident s and p partial
        waves) so the normalization is exact: returned shape broadcasts as ``[A, 1, 1]``.
        """
        forward = a_s * self.a_inc[..., :, _S_FWD] + a_p * self.a_inc[..., :, _P_FWD]
        return _poynting_z(forward)

    def _interface_fields(self, c_exit: np.ndarray) -> list[np.ndarray]:
        """Tangential field vector ``Gᵢ`` at the top interface of every layer.

        Walks up the stack from the exit using the stored layer matrices:
        ``G_exit = M_exit · c_exit`` then ``Gᵢ = Mᵢ · G_{i+1}``. Index 0 (prism) is
        unused and returned as ``None``; entries ``1 … n-1`` are populated, each
        broadcasting to ``[A, B, F, 4]``.
        """
        fields: list[np.ndarray] = [None] * self.n_layers
        last = self.n_layers - 1
        fields[last] = _matvec(self.layers[last].matrix, c_exit)
        for i in range(last - 1, 0, -1):
            fields[i] = _matvec(self.layers[i].matrix, fields[i + 1])
        return fields

    def _solve(self, polarization: str | tuple[complex, complex]):
        """Common front half: amplitudes, incident flux, and per-interface fields."""
        a_s, a_p = self._resolve_polarization(polarization)
        c_exit = self._transmitted_amplitudes(a_s, a_p)
        s_inc = self._incident_flux(a_s, a_p)
        fields = self._interface_fields(c_exit)
        return a_s, a_p, c_exit, s_inc, fields

    # -- power quantities -----------------------------------------------------

    def reflectance(self, polarization: str | tuple[complex, complex] = "p") -> np.ndarray:
        """Total power reflectance ``R`` for the given incident polarization.

        ``R = 1 − S_z(G₁)/S_z^inc``. For pure p/s incidence this equals
        ``|r_pp|²+|r_sp|²`` / ``|r_ss|²+|r_ps|²`` (a useful normalization check).
        Returned in the scenario's presentation shape (scalar for ``Simple``).
        """
        _, _, _, s_inc, fields = self._solve(polarization)
        return present(1.0 - _poynting_z(fields[1]) / s_inc)

    def transmittance(self, polarization: str | tuple[complex, complex] = "p") -> np.ndarray:
        """Power transmittance ``T`` -- the flux crossing into the exit medium.

        ``T = S_z(G_exit)/S_z^inc``. For a transparent substrate this is the power
        that propagates away; for a lossy semi-infinite exit it is the power
        delivered into (and ultimately absorbed by) the bulk -- see
        :meth:`field_profile` for the depth distribution. Presentation shape.
        """
        _, _, _, s_inc, fields = self._solve(polarization)
        return present(_poynting_z(fields[-1]) / s_inc)

    def transmission_coefficients(self) -> dict[str, np.ndarray]:
        """Amplitude transmission coefficients ``t_pp, t_ss, t_ps, t_sp``.

        These are the field-amplitude analogues of ``r_pp …`` -- exactly the
        coefficients pyGTM exposes from ``calculate_r_t`` and the quantities the
        field reconstruction is built on. They are the ``(s, p)`` forward
        transmission matrix ``T = M2⁻¹`` (``M2`` = the forward sub-block of ``Γ``),
        so ``c_exit = T · [a_s, a_p]``. Naming follows the package convention
        ``t_{in→out}`` (matching ``r_*``): e.g. ``t_ps`` is the s wave transmitted
        by incident p. They equal ``Structure.calculate_transmissivity`` exactly;
        ``transmittance`` then converts |amplitude|² to *power* with the correct
        Poynting/impedance weighting. Presentation shape.
        """
        g = self.gamma
        row0 = np.stack([g[..., _S_FWD, _S_FWD], g[..., _S_FWD, _P_FWD]], axis=-1)
        row1 = np.stack([g[..., _P_FWD, _S_FWD], g[..., _P_FWD, _P_FWD]], axis=-1)
        m2 = np.stack([row0, row1], axis=-2)  # [A, B, F, 2, 2]; rows/cols = (s, p)
        t = np.linalg.inv(m2)  # T[out, in]: row = transmitted (s, p), col = incident (s, p)
        return {
            "t_ss": present(t[..., 0, 0]),
            "t_ps": present(t[..., 0, 1]),  # in p -> out s
            "t_sp": present(t[..., 1, 0]),  # in s -> out p
            "t_pp": present(t[..., 1, 1]),
        }

    def _layer_absorption_canonical(self, s_inc: np.ndarray, fields: list[np.ndarray]):
        """Per-interior-layer absorption in canonical ``[A, B, F]`` form.

        Returns a list of ``(layer_index, layer_type, A_canonical)`` for the finite
        interior layers (indices ``1 … n-2``); empty when the stack is just
        prism + semi-infinite exit (the bulk case).
        """
        out = []
        for i in range(1, self.n_layers - 1):
            absorbed = (_poynting_z(fields[i]) - _poynting_z(fields[i + 1])) / s_inc
            out.append((i, self.layers[i].type, absorbed))
        return out

    def layer_absorption(
        self, polarization: str | tuple[complex, complex] = "p"
    ) -> list[dict[str, Any]]:
        """Layer-resolved absorptance for each finite interior layer.

        Returns:
            A list (top→bottom) of dicts ``{"index", "type", "absorptance"}``, where
            ``absorptance`` is the fraction of incident power dissipated in that
            layer, in presentation shape. Empty for a prism + semi-infinite-exit
            stack (no interior layers -- all absorbed power shows up as ``T`` into
            the bulk; use :meth:`field_profile` to resolve it with depth).
        """
        _, _, _, s_inc, fields = self._solve(polarization)
        return [
            {"index": i, "type": layer_type, "absorptance": present(absorbed)}
            for i, layer_type, absorbed in self._layer_absorption_canonical(s_inc, fields)
        ]

    def summary(self, polarization: str | tuple[complex, complex] = "p") -> dict[str, Any]:
        """One-call ``R``, ``T``, per-layer absorption, total, and conservation residual.

        ``conservation_residual = max|R + T + ΣAᵢ − 1|`` over the batch; it should
        be ~machine-epsilon for a correct, energy-conserving result.
        """
        _, _, _, s_inc, fields = self._solve(polarization)
        r = 1.0 - _poynting_z(fields[1]) / s_inc
        t = _poynting_z(fields[-1]) / s_inc
        per_layer = self._layer_absorption_canonical(s_inc, fields)
        total_abs = sum((a for _, _, a in per_layer), start=np.zeros_like(r))
        residual = np.abs(r + t + total_abs - 1.0)
        return {
            "R": present(r),
            "T": present(t),
            "layers": [
                {"index": i, "type": ty, "absorptance": present(a)} for i, ty, a in per_layer
            ],
            "total_absorption": present(total_abs),
            "conservation_residual": float(np.max(residual)),
        }

    def check_conservation(self, polarization: str | tuple[complex, complex] = "p") -> float:
        """Return ``max|R + T + ΣAᵢ − 1|`` over the batch (≈ 0 when correct)."""
        return self.summary(polarization)["conservation_residual"]

    # -- polarization-resolved power (experimental) ---------------------------

    def polarization_resolved(self, polarization: str = "p") -> dict[str, np.ndarray]:
        """**Experimental.** Split R and T into co- and cross-polarized channels.

        For a pure ``"p"`` or ``"s"`` incidence, the reflected/transmitted power is
        decomposed into the *co-polarized* channel (same polarization as incident)
        and the *cross-polarized* channel (the polarization-converted light — the
        power analogue of ``r_ps`` / ``t_ps``). The conversion fractions
        ``cross / (co + cross)`` quantify how much light changed polarization.

        Reflection is split exactly (``R_co = |r_co|²``, ``R_cross = |r_cross|²``,
        reflection being back into the same prism). Transmission is split via the
        per-mode flux of the two forward exit waves.

        Experimental / caveat: the transmitted s/p split is rigorous only for an
        **isotropic exit** medium, where the two exit eigenmodes are clean s and p
        and their fluxes add (``T_co + T_cross = T``). For an anisotropic exit the
        eigenmodes are elliptical, so the split is an eigenmode-resolved
        approximation and may not sum exactly to the total transmittance.

        Args:
            polarization: ``"p"`` or ``"s"`` (co/cross is undefined for a mixed
                Jones state).

        Returns:
            Dict (presentation shape) with ``R_co, R_cross, T_co, T_cross`` and
            ``conversion_reflection``, ``conversion_transmission``.

        Raises:
            ValueError: If ``polarization`` is not ``"p"`` or ``"s"``.
        """
        key = polarization.lower() if isinstance(polarization, str) else None
        if key not in ("p", "s"):
            raise ValueError("polarization_resolved requires pure 'p' or 's' incidence.")
        _, _, c_exit, s_inc, _ = self._solve(polarization)

        # Reflected amplitudes are the backward slots of c_inc = Gamma @ c_exit.
        c_inc = _matvec(self.gamma, c_exit)
        refl_s, refl_p = c_inc[..., _S_BWD], c_inc[..., _P_BWD]

        # Per-mode transmitted flux (exit modes: index 0 = s-like, 1 = p-like).
        w_full, _, amps = self._exit_transmitted_modes(c_exit)
        tangential = w_full[..., [0, 1, 3, 4], :]  # [..., (Ex,Ey,Hx,Hy), 2]

        def mode_transmittance(mode: int) -> np.ndarray:
            field = amps[..., mode][..., np.newaxis] * tangential[..., :, mode]
            return _poynting_z(field) / s_inc

        t_s, t_p = mode_transmittance(0), mode_transmittance(1)

        if key == "p":
            r_co, r_cross, t_co, t_cross = refl_p, refl_s, t_p, t_s
        else:
            r_co, r_cross, t_co, t_cross = refl_s, refl_p, t_s, t_p

        r_co_pow = np.abs(r_co) ** 2
        r_cross_pow = np.abs(r_cross) ** 2

        def _fraction(cross: np.ndarray, co: np.ndarray) -> np.ndarray:
            total = co + cross
            return np.where(np.abs(total) > 1e-30, cross / total, 0.0)

        return {
            "R_co": present(r_co_pow),
            "R_cross": present(r_cross_pow),
            "T_co": present(t_co),
            "T_cross": present(t_cross),
            "conversion_reflection": present(_fraction(r_cross_pow, r_co_pow)),
            "conversion_transmission": present(_fraction(t_cross, t_co)),
        }

    # -- field profile --------------------------------------------------------

    def _exit_transmitted_modes(
        self, c_exit: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full-field modes, ``kz`` and amplitudes of the two forward exit waves.

        Returns ``(W, kz, amps)`` with ``W`` shaped ``[..., 6, 2]`` (rows
        ``[Ex, Ey, Ez, Hx, Hy, Hz]``), ``kz`` ``[..., 2]`` and ``amps`` ``[..., 2]``
        ``= (c_exit[s_fwd], c_exit[p_fwd])``. Handles both a semi-infinite crystal
        exit (modes from its :class:`WaveProfile`) and an isotropic exit (modes from
        the exit dynamical matrix columns, with ``Ez``/``Hz`` from isotropic Maxwell).
        """
        exit_layer = self.layers[-1]
        amps = np.stack([c_exit[..., _S_FWD], c_exit[..., _P_FWD]], axis=-1)

        if getattr(exit_layer, "profile", None) is not None:
            w_full, kz = exit_layer.profile.full_modes()
            return w_full[..., :2], kz[..., :2], amps

        # Isotropic semi-infinite exit: tangential modes are the forward columns of
        # the exit dynamical matrix; recover Ez, Hz from isotropic Maxwell.
        m_exit = exit_layer.matrix  # [A, 1, 1, 4, 4]
        tangential = np.stack(
            [m_exit[..., :, _S_FWD], m_exit[..., :, _P_FWD]], axis=-1
        )  # [...,4,2]
        ex, ey, hx, hy = (tangential[..., j, :] for j in range(4))
        eps = np.complex128(exit_layer.eps_exit)
        kx = self.k_x[..., np.newaxis]  # [A, 1, 1, 1] -> trailing mode axis
        ez = -(kx / eps) * hy
        hz = kx * ey
        w_full = np.stack([ex, ey, ez, hx, hy, hz], axis=-2)  # [..., 6, 2]
        kz_scalar = np.sqrt(eps - self.k_x**2)  # forward branch (Im >= 0 by convention)
        kz = np.broadcast_to(kz_scalar[..., np.newaxis], amps.shape).copy()
        return w_full, kz, amps

    def _auto_semi_infinite_thickness(self, kz_exit: np.ndarray) -> float:
        """Pick a display window (cm) for the semi-infinite exit from mode decay.

        Uses ~5 amplitude e-foldings of the slowest-decaying transmitted mode; if
        all exit modes are propagating (lossless), falls back to a few free-space
        wavelengths so the standing/propagating field is still visible.
        """
        k0 = np.real(self.k_0).reshape(-1)
        rate = np.abs(np.imag(kz_exit)) * np.real(self.k_0)[..., np.newaxis]  # amplitude decay rate
        positive = rate[rate > 1e-9]
        if positive.size:
            return float(5.0 / positive.min())
        wavelength = 2.0 * np.pi / float(np.max(k0))  # = 1/freq (cm); shortest wavelength
        return 4.0 * wavelength

    def _reconstruct(
        self, w_full: np.ndarray, kz: np.ndarray, amps: np.ndarray, depth: np.ndarray
    ) -> np.ndarray:
        """Reconstruct full fields at depths ``depth`` (cm) below a layer's top.

        Each mode propagates as ``exp(+i·kz·k0·d)`` measured downward from the top
        interface, so forward (transmitted) modes decay into the substrate. Returns
        ``[..., Nz, 6]`` (rows ``[Ex, Ey, Ez, Hx, Hy, Hz]``).
        """
        # phase[..., z, mode] = exp(i kz k0 d)
        k0_d = self.k_0[..., np.newaxis] * depth  # [1, 1, F, Nz]
        phase = np.exp(1j * kz[..., np.newaxis, :] * k0_d[..., :, np.newaxis])  # [..., Nz, modes]
        weighted = phase * amps[..., np.newaxis, :]  # [..., Nz, modes]
        # field[..., z, comp] = sum_modes W[..., comp, mode] * weighted[..., z, mode]
        field = np.matmul(w_full[..., np.newaxis, :, :], weighted[..., np.newaxis])[..., 0]
        return field

    def field_profile(
        self,
        polarization: str | tuple[complex, complex] = "p",
        n_points: int = 200,
        semi_inf_thickness: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Reconstruct ``E(z), H(z)`` and ``S_z(z)`` through the whole stack.

        Args:
            polarization: ``"p"``, ``"s"`` or a complex ``(a_s, a_p)`` Jones pair.
            n_points: Depth samples per layer (interior layers and the exit window).
            semi_inf_thickness: Display window for the semi-infinite exit, in **µm**.
                ``None`` auto-selects ~5 decay lengths of the slowest transmitted
                mode (a few wavelengths if lossless).

        Returns:
            Dict with ``z`` (µm, measured from the first interface) and the squeezed
            field arrays ``Ex, Ey, Ez, Hx, Hy, Hz`` (complex) plus ``Sz`` (real,
            normalized to incident flux). Also ``absorption_cumulative`` =
            ``(Sz[0] − Sz)/1`` running from the surface and ``layer_boundaries`` (µm).

        Note:
            Intended for ``Simple`` / single-point use (the depth axis multiplies
            the batch size); see the class docstring.

        Raises:
            ValueError: If a layer thickness is being swept (canonical ``T`` axis
                size > 1) — the depth grid of the swept layer would itself change
                along ``T``, so a field-vs-depth profile is not well defined.
        """
        if self.gamma.shape[T] > 1:
            raise ValueError(
                "field_profile is undefined while sweeping a layer thickness "
                f"(T={self.gamma.shape[T]}); the depth grid changes along T. Profile a "
                "single thickness, or use the power quantities (reflectance/"
                "transmittance/layer_absorption), which support the T axis."
            )
        _, _, c_exit, s_inc, fields = self._solve(polarization)
        cm_to_um = 1.0e4

        z_segments: list[np.ndarray] = []
        comp_segments: list[np.ndarray] = []  # each [..., Nz, 6]
        boundaries = [0.0]
        z_offset = 0.0  # cm

        # Interior finite layers: amplitudes from V^{-1} G_i over all four modes.
        for i in range(1, self.n_layers - 1):
            layer = self.layers[i]
            # T == 1 here (guarded above), so a swept layer's [1,1,1,1] thickness
            # still reduces to a single scalar depth.
            thickness = float(np.asarray(layer.thickness).reshape(-1)[0])
            v_tan, kz = layer.profile.tangential_modes()
            w_full, _ = layer.profile.full_modes()
            amps = np.linalg.solve(v_tan, fields[i][..., np.newaxis])[..., 0]  # [..., 4]
            depth = np.linspace(0.0, thickness, n_points)
            comp_segments.append(self._reconstruct(w_full, kz, amps, depth))
            z_segments.append((z_offset + depth) * cm_to_um)
            z_offset += thickness
            boundaries.append(z_offset * cm_to_um)

        # Semi-infinite exit: two transmitted modes over a display window.
        w_full, kz_exit, amps = self._exit_transmitted_modes(c_exit)
        if semi_inf_thickness is None:
            exit_len = self._auto_semi_infinite_thickness(kz_exit)
        else:
            exit_len = float(semi_inf_thickness) / cm_to_um
        depth = np.linspace(0.0, exit_len, n_points)
        comp_segments.append(self._reconstruct(w_full, kz_exit, amps, depth))
        z_segments.append((z_offset + depth) * cm_to_um)
        boundaries.append((z_offset + exit_len) * cm_to_um)

        components = np.concatenate(comp_segments, axis=-2)  # [..., Nz_total, 6]
        z = np.concatenate(z_segments, axis=-1)
        sz = _poynting_z(components[..., [0, 1, 3, 4]]) / s_inc[..., np.newaxis]

        labels = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        result: dict[str, np.ndarray] = {"z": z, "layer_boundaries": np.asarray(boundaries)}
        for j, label in enumerate(labels):
            result[label] = np.squeeze(components[..., j])
        sz_sq = np.squeeze(sz)
        result["Sz"] = sz_sq
        result["absorption_cumulative"] = np.take(sz_sq, 0, axis=-1)[..., np.newaxis] - sz_sq
        return result

    def stokes_from_field_profile(
        self,
        polarization: str | tuple[complex, complex] = "p",
        n_points: int = 200,
        semi_inf_thickness: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Polarization state of the transverse field vs depth (Stokes + ellipse).

        Reconstructs the field with :meth:`field_profile` and forms the Stokes
        parameters of the transverse pair ``(Ex, Ey)`` at every depth, so you can
        watch the polarization ellipse evolve as the wave crosses a birefringent
        layer. ``Ex`` is the in-plane (p-like) component, ``Ey`` the out-of-plane
        (s-like) one; the convention is ``e^{-iωt}``.

        Args:
            polarization: ``"p"``, ``"s"`` or a complex ``(a_s, a_p)`` Jones pair.
            n_points: Depth samples per layer (passed to :meth:`field_profile`).
            semi_inf_thickness: Semi-infinite exit display window in µm (see
                :meth:`field_profile`).

        Returns:
            Dict with ``z`` (µm) and ``layer_boundaries`` plus the Stokes profiles
            ``S0`` (intensity) and ``S1, S2, S3``, and the ellipse parameters
            ``azimuth`` (ψ, rad) and ``ellipticity`` (χ, rad). A single coherent
            field is fully polarized, so the degree of polarization is 1 by
            construction and is not returned.

        Note:
            Like :meth:`field_profile`, this is undefined while sweeping a layer
            thickness (raises if the canonical ``T`` axis size > 1).
        """
        profile = self.field_profile(polarization, n_points, semi_inf_thickness)
        ex, ey = profile["Ex"], profile["Ey"]
        ex_sq, ey_sq = np.abs(ex) ** 2, np.abs(ey) ** 2
        s0 = ex_sq + ey_sq
        s1 = ex_sq - ey_sq
        s2 = 2.0 * np.real(ex * np.conj(ey))
        s3 = -2.0 * np.imag(ex * np.conj(ey))
        return {
            "z": profile["z"],
            "layer_boundaries": profile["layer_boundaries"],
            "S0": s0,
            "S1": s1,
            "S2": s2,
            "S3": s3,
            "azimuth": 0.5 * np.arctan2(s2, s1),
            "ellipticity": 0.5 * np.arctan2(s3, np.sqrt(s1**2 + s2**2)),
        }
