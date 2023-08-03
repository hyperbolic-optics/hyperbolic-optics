import json


def mock_incident_payload():
    payload = json.dumps({
        "ScenarioData": {
        "type": "Dispersion",
        "frequency": 1470,
    },
    "Layers": [
        {
            "type": "Ambient Incident Layer",
            "permittivity": 5.5
        },
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Calcite",
            "rotationX": 0,
            "rotationY": 67,
            "rotationZ": 0,
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