import json

def mock_simple_payload():
    """
    Create a simple scenario payload for testing.
    This scenario uses:
    - Single incident angle (30 degrees)
    - Single azimuthal angle (45 degrees) 
    - Single frequency (1460 cm^-1)
    - Single layer orientation
    """
    payload = json.dumps({
        "ScenarioData": {
            "type": "Simple",
            "incidentAngle": 45.0,        # degrees
            "azimuthal_angle": 90.0,      # degrees (optional, defaults to 0)
            "frequency": 460           # cm^-1
        },
        "Layers": [
            {
                "type": "Ambient Incident Layer",
                "permittivity": 50.
            },
            {
                "type": "Isotropic Middle-Stack Layer",
                "thickness": 0.1,
                "permittivity": 1.0
            },
            # {
            #     "type": "Crystal Layer",
            #     "material": "Calcite",
            #     "rotationX": 0,
            #     "rotationY": 90,
            #     "rotationZ": 0.0,
            #     "thickness": 3.0
            # },
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": "Quartz",
                "rotationX": 0,
                "rotationY": 90,
                "rotationZ": 0
            }
        ]
    })
    return payload

def mock_simple_dielectric_payload():
    """
    Create a simple scenario payload using arbitrary dielectric material.
    """
    payload = json.dumps({
        "ScenarioData": {
            "type": "Simple",
            "incidentAngle": 45.0,
            "azimuthal_angle": 0.0,
            "frequency": 1460.0
        },
        "Layers": [
            {
                "type": "Ambient Incident Layer",
                "permittivity": 22.5
            },
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": {
                    "eps_xx": {"real": 2.2652, "imag": 0.00065},
                    "eps_yy": {"real": -4.83671, "imag": 0.75521}, 
                    "eps_zz": {"real": -4.83671, "imag": 0.75521},
                    "eps_xy": {"real": 0.0, "imag": 0.0},
                    "eps_xz": {"real": 0.0, "imag": 0.0},
                    "eps_yz": {"real": 0.0, "imag": 0.0}
                },
                "rotationX": 0,
                "rotationY": 0,
                "rotationZ": 0.0
            }
        ]
    })
    return payload

def mock_incident_payload():
    payload = json.dumps({
        "ScenarioData": {
        "type": "Incident",
    },
    "Layers": [
        {
            "type": "Ambient Incident Layer",
            "permittivity": 12.5
        },
        {
            "type": "Isotropic Middle-Stack Layer",
            "thickness": 23.
        },
        {
            "type": "Crystal Layer",
            "material": "MnF2",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 180.,
            "thickness": 30.,
        },
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "MnF2",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0,
        }
    ],
    })
    return payload


def mock_azimuthal_payload():
    payload = json.dumps({
        "ScenarioData": {
        "type": "Azimuthal",
        "incidentAngle": 30,
    },
    "Layers": [
        {
            "type": "Ambient Incident Layer",
            "permittivity": 12.5
        },
        {
            "type": "Isotropic Middle-Stack Layer",
            "thickness": 23.
        },
        {
            "type": "Crystal Layer",
            "material": "MnF2",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 180.,
            "thickness": 30.,
        },
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "MnF2",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0,
        }
    ],
    })

    return payload


def mock_dispersion_payload():
    payload = json.dumps({
    "ScenarioData": {
        "type": "Dispersion",
        "frequency": 1460
    },
    "Layers": [
        {
                "type": "Ambient Incident Layer",
                "permittivity": 50.
            },
            {
                "type": "Isotropic Middle-Stack Layer",
                "thickness": 0.5,
                "permittivity": 1.0
            },
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": "Calcite",
                "rotationX": 0,
                "rotationY": 90,
                "rotationZ": 0
            },
        ],
    })


    return payload



def updating_payload(scenario, material, eps_prism, air_gap_thickness, rotationY, rotationZ, incident_angle, frequency):

    payload = {}

    if scenario == "Incident":
        payload["ScenarioData"] = {
            "type": scenario
        }
    elif scenario == "Azimuthal":
        payload["ScenarioData"] = {
            "type": scenario,
            "incidentAngle": incident_angle
        }
    elif scenario == "Dispersion":
        payload["ScenarioData"] = {
            "type": scenario,
            "frequency": frequency
        }
    
    payload["Layers"] = [
        {
            "type": "Ambient Incident Layer",
            "permittivity": eps_prism
        },
        {
            "type": "Isotropic Middle-Stack Layer",
            "thickness": air_gap_thickness,
            "permittivity": 1.
        },
    ]

    bulk_layer = {
        "type": "Semi Infinite Anisotropic Layer",
        "material": material,
        "rotationX": 0.,
        "rotationY": rotationY,
        "rotationZ": rotationZ,
    }

    payload["Layers"].append(bulk_layer)

    return json.dumps(payload)