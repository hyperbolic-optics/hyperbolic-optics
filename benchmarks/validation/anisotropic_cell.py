#!/usr/bin/env python3
"""Independent NGSolve FEM / RCWA comparison for one rotated biaxial periodic cell.

Run FEM in a separate environment: uv run --no-project --with ngsolve --with numpy
    benchmarks/validation/anisotropic_cell.py fem --output /tmp/fem.json
Run RCWA: uv run python benchmarks/validation/anisotropic_cell.py rcwa --output /tmp/rcwa.json
Units µm; exp(-iωt); air on both sides; reference planes z=0,h.
The FEM code imports no hyperbolic_optics code, material model or Fourier operators.
"""

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

PITCH, HEIGHT, WAVELENGTH = 0.6, 0.3, 1.55
SIZE = (0.3, 0.2)


def geometry(case):
    if case == "hbn":
        wavelength = 1e4 / 1560.0
        scale = wavelength / WAVELENGTH
        return PITCH * scale, 0.1 * scale, wavelength, np.array(SIZE) * scale
    return PITCH, HEIGHT, WAVELENGTH, np.array(SIZE)


def tensor(case="dielectric"):
    """Independent Rz Ry Rx rotation; hBN Lorentz parameters are explicit here."""
    x, y, z = np.radians([17.0, 31.0, 23.0])
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    rotation = (
        np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    )
    principal = [2.25, 3.24, 4.0]
    if case == "hbn":
        # Caldwell et al., Nat. Commun. 5, 5221 (2014); cm^-1, exp(-iωt).
        def lorentz(einf, to, lo, gamma):
            return einf * (1 + (lo**2 - to**2) / (to**2 - 1560.0**2 - 1j * gamma * 1560.0))

        ordinary = lorentz(4.9, 1360.0, 1614.0, 7.0)
        principal = [ordinary, ordinary, lorentz(2.95, 760.0, 825.0, 2.0)]
    return rotation @ np.diag(principal) @ rotation.T


def fem(
    maxh, padding=1.5, control=None, crystal_mesh=None, *, case="dielectric", angle=0.0, azimuth=0.0
):
    import ngsolve as ng
    from netgen.occ import Box, Glue, OCCGeometry, Pnt, Z

    ng.SetNumThreads(2)
    pitch, height, wavelength, size = geometry(case)
    # Scale coordinates for comparable mesh settings between infrared and visible cases.
    unit = wavelength / WAVELENGTH
    pitch, height, wavelength, size = pitch / unit, height / unit, wavelength / unit, size / unit
    k = 2 * np.pi / wavelength
    theta, phi = np.radians([angle, azimuth])
    cosine = np.cos(theta)
    p = ng.CF((np.cos(phi), np.sin(phi), 0.0))
    s = ng.CF((-np.sin(phi), np.cos(phi), 0.0))
    q = k * np.sin(theta) * p
    parts = []
    edges = [-padding, -0.5, -0.2, 0.0, height, height + 0.2, height + 0.5, height + padding]
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        box = Box(Pnt(-pitch / 2, -pitch / 2, low), Pnt(pitch / 2, pitch / 2, high))
        box.mat("air")
        if low == -padding:
            box.faces.Min(Z).name = "input"
        if high == height + padding:
            box.faces.Max(Z).name = "output"
        if high == -0.2:
            box.faces.Max(Z).name = "probe_top"
        if low == height + 0.2:
            box.faces.Min(Z).name = "probe_bottom"
        if low == 0.0:
            if control:
                box.mat("crystal")
                parts.append(box)
            else:
                pillar = Box(
                    Pnt(-size[0] / 2, -size[1] / 2, 0), Pnt(size[0] / 2, size[1] / 2, height)
                )
                pillar.mat("crystal")
                if crystal_mesh is not None:
                    pillar.maxh = crystal_mesh
                    pillar.faces.maxh = crystal_mesh
                parts.extend([box - pillar, pillar])
        else:
            parts.append(box)
    shape = Glue(parts)
    for axis in (0, 1):
        left = [f for f in shape.faces if abs(f.center[axis] + pitch / 2) < 1e-8]
        right = [f for f in shape.faces if abs(f.center[axis] - pitch / 2) < 1e-8]
        for i, face in enumerate(left):
            other = min(
                right,
                key=lambda f: sum(
                    (f.center[j] - face.center[j]) ** 2 for j in range(3) if j != axis
                ),
            )
            other.Identify(face, f"periodic{axis}_{i}")
    mesh = ng.Mesh(OCCGeometry(shape).GenerateMesh(maxh=maxh))
    space = ng.Periodic(ng.HCurl(mesh, order=2, complex=True))
    print(
        dict(
            stage="FEM assembly",
            dofs=space.ndof,
            maxh=maxh,
            crystal_mesh=crystal_mesh,
            padding=padding,
            control=control,
        ),
        flush=True,
    )
    u, v = space.TnT()
    eps = np.eye(3) * (1.0 if control == "empty" else 4.0) if control else tensor(case)
    crystal = ng.CF(tuple(eps.ravel()), dims=(3, 3))
    identity = ng.CF(tuple(np.eye(3).ravel()), dims=(3, 3))
    epsilon = mesh.MaterialCF({"crystal": crystal}, default=identity)
    form = ng.BilinearForm(space)
    # E = exp(i q·r) u with periodic u. Conjugating the test field reverses i q.
    form += (
        (ng.curl(u) + 1j * ng.Cross(q, u)) * (ng.curl(v) - 1j * ng.Cross(q, v))
        - k * k * (epsilon * u) * v
    ) * ng.dx

    # Exact outgoing impedance for the only propagating (zeroth) order.
    # Other orders decay in the vacuum spacer; vary padding to check their reflection.
    def admittance(a, b):
        return (a * p) * (b * p) / cosine + cosine * (a * s) * (b * s)

    form += -1j * k * admittance(u.Trace(), v.Trace()) * ng.ds("input|output")
    form.Assemble()
    inverse = form.mat.Inverse(space.FreeDofs(), inverse="umfpack")
    reflection, transmission, fields = [], [], []
    for pol in (cosine * p + ng.CF((0, 0, -np.sin(theta))), s):
        incident = pol * ng.exp(1j * k * cosine * ng.z)
        rhs = ng.LinearForm(space)
        rhs += -2j * k * admittance(incident, v.Trace()) * ng.ds("input")
        rhs.Assemble()
        field = ng.GridFunction(space)
        field.vec.data = inverse * rhs.vec

        def average(cf, boundary):
            return (
                np.array(
                    [
                        ng.Integrate(cf * direction, mesh, definedon=mesh.Boundaries(boundary))
                        for direction in (p / cosine, s)
                    ]
                )
                / pitch**2
            )

        r = average(field - incident, "probe_top") * np.exp(-1j * k * cosine * 0.2)
        r[0] *= -1  # reflected p shares Hy's sign, so Ex is negative
        reflection.append(r)
        transmission.append(average(field, "probe_bottom") * np.exp(-1j * k * cosine * 0.2))
        fields.append(field)
    loss = ng.CF(tuple(np.asarray(eps).imag.ravel()), dims=(3, 3))
    absorption = np.array(
        [
            [
                k
                / (cosine * pitch**2)
                * ng.Integrate(ng.Conj(a) * (loss * b), mesh, definedon=mesh.Materials("crystal"))
                for b in fields
            ]
            for a in fields
        ]
    )
    return (
        np.array(reflection).T,
        np.array(transmission).T,
        dict(
            maxh=maxh,
            padding=padding,
            length_unit_um=unit,
            dofs=space.ndof,
            element_order=2,
            control=control,
            crystal_mesh=crystal_mesh,
            volume_absorption_real=absorption.real.tolist(),
            volume_absorption_imag=absorption.imag.tolist(),
        ),
    )


def rcwa(order, forward_only=False, *, case="dielectric", angle=0.0, azimuth=0.0):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from hyperbolic_optics import rcwa2d

    pitch, height, wavelength, size = geometry(case)

    def scattering(width):
        layers = [
            rcwa2d.PatternedLayer(
                height,
                jnp.eye(3),
                (
                    rcwa2d.Shape(
                        rcwa2d.rectangle((0.0, 0.0), (size[0], width)), jnp.asarray(tensor(case))
                    ),
                ),
            )
        ]
        return rcwa2d.crossed_grating_smatrix(
            layers,
            (pitch, pitch),
            wavelength,
            np.radians(angle),
            1.0,
            1.0,
            (order, order),
            azimuth_rad=np.radians(azimuth),
        ).s

    solve = jax.jit(scattering)
    s = np.asarray(solve(size[1]))
    if forward_only:
        jax.clear_caches()
        return s[:2, :2], s[2:, :2], dict(order=order, harmonics=(2 * order + 1) ** 2)
    objective = jax.jit(lambda width: jnp.abs(scattering(width)[3, 0]) ** 2)
    gradient = float(jax.jit(jax.grad(objective))(size[1]))
    finite_difference = {
        str(step): float((objective(size[1] + step) - objective(size[1] - step)) / (2 * step))
        for step in (1e-3, 1e-4, 1e-5)
    }
    jax.clear_caches()
    return (
        s[:2, :2],
        s[2:, :2],
        dict(
            order=order,
            harmonics=(2 * order + 1) ** 2,
            gradient=gradient,
            finite_difference=finite_difference,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solver", choices=("fem", "rcwa"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--case", choices=("dielectric", "hbn"), default="dielectric")
    parser.add_argument("--angle", type=float, default=0.0, help="Incidence angle in degrees")
    parser.add_argument(
        "--azimuth", type=float, default=0.0, help="Incidence-plane azimuth in degrees"
    )
    parser.add_argument("--meshes", type=float, nargs="+", default=[0.16, 0.12, 0.09, 0.08])
    parser.add_argument(
        "--crystal-meshes",
        type=float,
        nargs="+",
        help="Local crystal refinement with the finest --meshes vacuum mesh",
    )
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()
    controls = [(0.09, 1.5, "empty"), (0.09, 1.5, "film")]
    runs = (
        (
            controls
            if args.controls_only
            else [(h, 1.5, None) for h in args.meshes] + [(min(args.meshes), 1.0, None)] + controls
        )
        if args.solver == "fem"
        else [(m, args.forward_only) for m in args.orders]
    )
    if args.solver == "fem" and args.crystal_meshes and not args.controls_only:
        runs = [(min(args.meshes), 1.5, None, h) for h in args.crystal_meshes]
        runs += [(min(args.meshes), 1.0, None, min(args.crystal_meshes))] + controls
    pitch, height, wavelength, size = geometry(args.case)
    epsilon = tensor(args.case)
    data = dict(
        solver=args.solver,
        version=version("ngsolve" if args.solver == "fem" else "jax"),
        case=args.case,
        rotation_deg=[17.0, 31.0, 23.0],
        angle_deg=args.angle,
        azimuth_deg=args.azimuth,
        pitch_um=pitch,
        height_um=height,
        wavelength_um=wavelength,
        size_um=size.tolist(),
        epsilon_real=epsilon.real.tolist(),
        epsilon_imag=epsilon.imag.tolist(),
        runs=[],
    )
    for params in runs:
        start = time.perf_counter()
        r, t, row = (fem if args.solver == "fem" else rcwa)(
            *params, case=args.case, angle=args.angle, azimuth=args.azimuth
        )
        absorption = np.eye(2) - r.conj().T @ r - t.conj().T @ t
        row.update(
            seconds=time.perf_counter() - start,
            absorption_real=absorption.real.tolist(),
            absorption_imag=absorption.imag.tolist(),
            r_real=r.real.tolist(),
            r_imag=r.imag.tolist(),
            t_real=t.real.tolist(),
            t_imag=t.imag.tolist(),
        )
        if args.solver == "fem":
            volume = np.array(row["volume_absorption_real"]) + 1j * np.array(
                row["volume_absorption_imag"]
            )
            row["balance_error"] = float(np.max(abs(absorption - volume)))
        if row.get("control"):
            n = 1 if row["control"] == "empty" else 2
            c = np.cos(np.radians(args.angle))
            kz = np.sqrt(n**2 - np.sin(np.radians(args.angle)) ** 2)
            phase = np.exp(2j * np.pi * kz * height / wavelength)
            air, film = np.array([1 / c, c]), np.array([n**2 / kz, kz])
            q = (air - film) / (air + film)
            expected_r = np.diag(q * (1 - phase**2) / (1 - q * q * phase**2) * [-1, 1])
            expected_t = np.diag((1 - q * q) * phase / (1 - q * q * phase**2))
            row["analytic_error"] = float(
                max(np.max(abs(r - expected_r)), np.max(abs(t - expected_t)))
            )
            assert row["analytic_error"] < 0.002, row
        data["runs"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2))
        print({k: v for k, v in row.items() if not k.endswith(("real", "imag"))}, flush=True)


if __name__ == "__main__":
    main()
