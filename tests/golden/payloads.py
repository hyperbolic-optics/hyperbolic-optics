"""Payload battery for the golden-master regression harness.

Each entry in :data:`PAYLOADS` is a complete ``Structure.execute`` payload. The
battery deliberately spans every scenario type, both ``rotationZType`` modes, the
arbitrary-material path, a magnetic air gap, an isotropic exit, the monoclinic
material path, and at least one hyperbolic ``kx/k0 > 1`` (evanescent/leaky) case.

Notes on construction (verified against the current code):

- For ``Incident`` / ``Azimuthal`` / ``FullSweep`` the frequency range is taken
  from the *last* layer's material (``structure.py`` ``get_layers``), and each
  material is evaluated over its *own* default range. Stacks that combine a
  finite ``Crystal Layer`` with a semi-infinite crystal therefore use the **same
  material** for both so the (length-410) frequency axes are physically coherent.
- A ``Semi Infinite Isotropic Layer`` exit only composes by broadcasting in the
  scalar-frequency scenarios (``Simple``/``Dispersion``); it is used here in a
  ``Simple`` stack.
- Thickness is in microns (converted to cm via ``*1e-4`` in ``layers.py``).
"""

from typing import Any

# --- reusable layer fragments -------------------------------------------------

_CALCITE_Y90 = {
    "type": "Semi Infinite Anisotropic Layer",
    "material": "Calcite",
    "rotationX": 0,
    "rotationY": 90,
    "rotationZ": 0,
}

_QUARTZ_Y90 = {
    "type": "Semi Infinite Anisotropic Layer",
    "material": "Quartz",
    "rotationX": 0,
    "rotationY": 90,
    "rotationZ": 0,
}


PAYLOADS: dict[str, dict[str, Any]] = {
    # 1. Simple: scalar path, baseline.
    "simple_calcite": {
        "ScenarioData": {
            "type": "Simple",
            "incidentAngle": 45.0,
            "azimuthal_angle": 0.0,
            "frequency": 1460.0,
        },
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 50.0},
            {"type": "Isotropic Middle-Stack Layer", "thickness": 0.1, "permittivity": 1.0},
            _CALCITE_Y90,
        ],
    },
    # 2. Incident: kx-omega sweep, (410, 360) output.
    "incident_calcite": {
        "ScenarioData": {"type": "Incident"},
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 12.5},
            {"type": "Isotropic Middle-Stack Layer", "thickness": 0.5},
            _CALCITE_Y90,
        ],
    },
    # 3. Azimuthal: the swapaxes axis-order fix point; rotationZType relative (default).
    "azimuthal_calcite": {
        "ScenarioData": {"type": "Azimuthal", "incidentAngle": 40},
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 12.5},
            {"type": "Isotropic Middle-Stack Layer", "thickness": 0.5},
            _CALCITE_Y90,
        ],
    },
    # 4. Dispersion: azimuth folded into rotationZ, (180, 480) output.
    "dispersion_calcite": {
        "ScenarioData": {"type": "Dispersion", "frequency": 1460.0},
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 25.0},
            {"type": "Isotropic Middle-Stack Layer", "thickness": 0.5, "permittivity": 1.0},
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": "Calcite",
                "rotationX": 0,
                "rotationY": 70,
                "rotationZ": 0,
            },
        ],
    },
    # 5. FullSweep: 3D (410, 180, 120) output; anisotropic exit only (no isotropic
    #    exit FullSweep branch yet). NOTE: heavy (180x120x410) -- marked slow in
    #    the test module.
    "fullsweep_quartz": {
        "ScenarioData": {"type": "FullSweep"},
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 50.0},
            _QUARTZ_Y90,
        ],
    },
    # 6. Arbitrary material: ArbitraryMaterial eps dict path.
    "arbitrary_material": {
        "ScenarioData": {
            "type": "Simple",
            "incidentAngle": 45.0,
            "azimuthal_angle": 0.0,
            "frequency": 1460.0,
        },
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 22.5},
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": {
                    "eps_xx": {"real": 2.27, "imag": 0.001},
                    "eps_yy": {"real": -4.84, "imag": 0.755},
                    "eps_zz": {"real": -4.84, "imag": 0.755},
                    "eps_xy": {"real": 0.0, "imag": 0.0},
                    "eps_xz": {"real": 0.0, "imag": 0.0},
                    "eps_yz": {"real": 0.0, "imag": 0.0},
                },
                "rotationX": 0,
                "rotationY": 0,
                "rotationZ": 0.0,
            },
        ],
    },
    # 7. Multi-layer: prism + air gap + finite Crystal Layer + semi-infinite,
    #    same material (Quartz) for crystal and exit to keep freq axes coherent.
    "multilayer_incident": {
        "ScenarioData": {"type": "Incident"},
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 50.0},
            {"type": "Isotropic Middle-Stack Layer", "thickness": 0.5, "permittivity": 1.0},
            {
                "type": "Crystal Layer",
                "material": "Quartz",
                "thickness": 1.0,
                "rotationX": 0,
                "rotationY": 70,
                "rotationZ": 0,
            },
            _QUARTZ_Y90,
        ],
    },
    # 8. Azimuthal with rotationZType "static" (vs relative in #3).
    "rotz_static_azimuthal": {
        "ScenarioData": {"type": "Azimuthal", "incidentAngle": 40},
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 12.5},
            {"type": "Isotropic Middle-Stack Layer", "thickness": 0.5},
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": "Calcite",
                "rotationX": 0,
                "rotationY": 90,
                "rotationZ": 30,
                "rotationZType": "static",
            },
        ],
    },
    # 9. Isotropic exit layer (Simple scenario, where it composes cleanly).
    "isotropic_exit_simple": {
        "ScenarioData": {
            "type": "Simple",
            "incidentAngle": 30.0,
            "azimuthal_angle": 0.0,
            "frequency": 1460.0,
        },
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 50.0},
            {
                "type": "Crystal Layer",
                "material": "Calcite",
                "thickness": 1.0,
                "rotationX": 0,
                "rotationY": 90,
                "rotationZ": 0,
            },
            {"type": "Semi Infinite Isotropic Layer", "permittivity": 1.0},
        ],
    },
    # 10. Magnetic air gap: exercises a non-trivial mu tensor through the pipeline.
    "magnetic_gap_incident": {
        "ScenarioData": {"type": "Incident"},
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 12.5},
            {
                "type": "Isotropic Middle-Stack Layer",
                "thickness": 0.5,
                "permittivity": 2.0,
                "permeability": {"real": 1.5, "imag": 0.1},
            },
            _CALCITE_Y90,
        ],
    },
    # 11. Hyperbolic kx/k0 > 1: high-permittivity prism into Calcite reststrahlen
    #     -> large kx, the evanescent/leaky regime Phase 3/4 must get right.
    "hyperbolic_evanescent": {
        "ScenarioData": {"type": "Incident"},
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 50.0},
            _CALCITE_Y90,
        ],
    },
    # 12. Monoclinic material (off-diagonal eps) path: Gallium Oxide.
    "gallium_oxide_incident": {
        "ScenarioData": {"type": "Incident"},
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 25.0},
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": "GalliumOxide",
                "rotationX": 0,
                "rotationY": 90,
                "rotationZ": 0,
            },
        ],
    },
}

# Payloads that are expensive to run (marked slow in the test module).
# --- materials and axes the battery previously never touched ------------------
#
# The library ships ten materials; before these, only Calcite, Quartz and
# GalliumOxide appeared here, so half of it had no value locked anywhere. That
# is how alpha-MoO3 kept two Reststrahlen bands on the wrong crystal axes and
# hBN kept a mixture of two references' parameters. A list-valued thickness (the
# canonical T axis) was likewise unrepresented.

PAYLOADS["moo3_incident"] = {
    "ScenarioData": {"type": "Incident", "polar_points": 40},
    "Layers": [
        {"type": "Ambient Incident Layer", "permittivity": 5.5},
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "MoO3",
            "rotationX": 0,
            "rotationY": 0,
            "rotationZ": 0,
        },
    ],
}

PAYLOADS["hbn_azimuthal"] = {
    "ScenarioData": {"type": "Azimuthal", "incidentAngle": 30.0, "azimuthal_points": 40},
    "Layers": [
        {"type": "Ambient Incident Layer", "permittivity": 11.56},
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "hBN",
            "rotationX": 0,
            "rotationY": 70,
            "rotationZ": 0,
        },
    ],
}

PAYLOADS["sic_aln_stack_incident"] = {
    "ScenarioData": {"type": "Incident", "polar_points": 40},
    "Layers": [
        {"type": "Ambient Incident Layer", "permittivity": 11.56},
        {"type": "Crystal Layer", "material": "AlN", "thickness": 1.0, "rotationY": 0},
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "SiC",
            "rotationX": 0,
            "rotationY": 0,
            "rotationZ": 0,
        },
    ],
}

PAYLOADS["gan_simple"] = {
    "ScenarioData": {
        "type": "Simple",
        "incidentAngle": 40.0,
        "azimuthal_angle": 25.0,
        "frequency": 600.0,
    },
    "Layers": [
        {"type": "Ambient Incident Layer", "permittivity": 11.56},
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "GaN",
            "rotationX": 0,
            "rotationY": 60,
            "rotationZ": 0,
        },
    ],
}

PAYLOADS["sapphire_calcite_lower_dispersion"] = {
    "ScenarioData": {
        "type": "Dispersion",
        "frequency": 600.0,
        "polar_points": 30,
        "azimuthal_points": 40,
    },
    "Layers": [
        {"type": "Ambient Incident Layer", "permittivity": 11.56},
        {"type": "Crystal Layer", "material": "CalciteLower", "thickness": 1.0, "rotationY": 45},
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Sapphire",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0,
        },
    ],
}

# List-valued thickness: locks the canonical T axis and the broadcast that
# builds it, which nothing else in the battery exercises.
PAYLOADS["thickness_axis_simple"] = {
    "ScenarioData": {
        "type": "Simple",
        "incidentAngle": 45.0,
        "azimuthal_angle": 0.0,
        "frequency": 1460.0,
    },
    "Layers": [
        {"type": "Ambient Incident Layer", "permittivity": 50.0},
        {
            "type": "Crystal Layer",
            "material": "Calcite",
            "thickness": [0.5, 1.0, 1.5, 2.0],
            "rotationY": 90,
        },
        _CALCITE_Y90,
    ],
}


SLOW_PAYLOADS = {"fullsweep_quartz"}
