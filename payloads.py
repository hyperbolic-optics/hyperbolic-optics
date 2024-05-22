import json


def mock_incident_payload():
    payload = json.dumps({
        "ScenarioData": {
        "type": "Incident",
    },
    "Layers": [
        {
            "type": "Ambient Incident Layer",
            "permittivity": 5.5
        },
        {
            "type": "Isotropic Middle-Stack Layer",
            "thickness": 1.5
        },
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Quartz",
            "rotationX": 0,
            "rotationY": 70,
            "rotationZ": 45,
        }
    ],
    })
    return payload


def mock_azimuthal_payload():
    payload = json.dumps({
        "ScenarioData": {
        "type": "Azimuthal",
        "incidentAngle": 50,
    },
    "Layers": [
        {
            "type": "Ambient Incident Layer",
            "permittivity": 50.5
        },
        {
            "type": "Isotropic Middle-Stack Layer",
            "thickness": 0.1
        },
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Quartz",
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
        "frequency": 475
    },
    "Layers": [
        {
            "type": "Ambient Incident Layer",
            "permittivity": 11.5
        },
        {
            "type": "Isotropic Middle-Stack Layer",
            "thickness": 1.5
        },
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Quartz",
            "rotationX": 0,
            "rotationY": 70,
            "rotationZ": 45,
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
        {
            "type": "Crystal Layer",
            "thickness": 0.1,
            "material": material,
            "rotationX": 0.,
            "rotationY": 66.7,
            "rotationZ": 180.,
            "rotationZType": "relative"
        }
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