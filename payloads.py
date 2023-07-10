import json


def mock_incident_payload():
    payload = json.dumps({
        "ScenarioData": {
        "type": "Incident",
    },
    "Layers": [
        {
            "type": "Ambient Incident Layer",
            "permittivity": 100.
        },
        {
            "type": "Isotropic Middle-Stack Layer",
            "thickness": 0.1
        },
        {
            "type": "Crystal Layer",
            "thickness": 2.,
            "material": "Quartz",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0,
        },
        {
            "type": "Semi Infinite Isotropic Layer",
            "permittivity": 1.
        }
    ],
    })
    return payload


def mock_azimuthal_payload():
    payload = json.dumps({
        "ScenarioData": {
        "type": "Azimuthal",
        "incidentAngle": 45,
    },
    "Layers": [
        {
            "type": "Ambient Incident Layer",
            "permittivity": 5.5
        },
        {
            "type": "Isotropic Middle-Stack Layer",
            "thickness": 0.1
        },
        {
            "type": "Crystal Layer",
            "thickness": 1.,
            "material": "Quartz",
            "rotationX": 0,
            "rotationY": 45,
            "rotationZ": 0,
        },
        {
            "type": "Semi Infinite Isotropic Layer",
            "permittivity": 1.
        }
    ],
    })

    return payload


def mock_dispersion_payload():
    payload = json.dumps({
    "ScenarioData": {
        "type": "Dispersion",
        "frequency": 475
    },
    "Layers": [
        {
            "type": "Ambient Incident Layer",
            "permittivity": 40.
        },
        {
            "type": "Crystal Layer",
            "thickness": 3.,
            "material": "Quartz",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0,
            "rotationZType": "relative"
        },
        {
            "type": "Semi Infinite Isotropic Layer",
            "permittivity": 1.
        }
    ],
    })

    return payload



def updating_payload(scenario, eps_prism, air_gap_thickness, rotationY, rotationZ, incident_angle, frequency):
    payload = json.dumps({
        "ScenarioData": {
        "type": scenario
    },
    "Layers": [
        {
            "type": "Ambient Incident Layer",
            "permittivity": eps_prism
        },
        {
            "type": "Isotropic Middle-Stack Layer",
            "thickness": air_gap_thickness,
            "permittivity": 1.
        },
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Quartz",
            "rotationX": 0,
            "rotationY": rotationY,
            "rotationZ": rotationZ,
        }
    ],
    })
    return payload