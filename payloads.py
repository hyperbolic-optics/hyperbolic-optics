import json


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
        "frequency": 53
    },
    "Layers": [
        {
            "type": "Ambient Incident Layer",
            "permittivity": 22.5
        },
        {
            "type": "Isotropic Middle-Stack Layer",
            "thickness": 10.
        },
        {
                "type": "Crystal Layer",
                "material": {
                    # Arbitrary dielectric material with custom permittivity tensor
                    "eps_xx": {"real": 2.2652, "imag": 0.00065},
                    "eps_yy": {"real": -4.83671, "imag": 0.75521}, 
                    "eps_zz": {"real": -4.83671, "imag": 0.75521},
                    # Magnetic permeability (optional, defaults to identity)
                    "mu_xx": 1.0,
                    "mu_yy": 1.0,
                    "mu_zz": 1.0,
                    "mu_xy": 0.0,
                    "mu_xz": 0.0,
                    "mu_yz": 0.0
                },
                "rotationX": 0,
                "rotationY": 0,
                "rotationZ": 0.,
                "thickness": 15.5,
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