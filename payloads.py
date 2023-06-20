import json


def mock_incident_payload():
    payload = json.dumps({
        "ScenarioData": {
        "type": "Incident",
        "incidentAngle": {"min": -90, "max": 90},
    },
    "Layers": [
        {
            "type": "Ambient Incident Layer",
            "permittivity": 5.5,
        },
        {
            "type": "Isotropic Middle-Stack Layer",
            "permittivity": 1,
            "thickness": 1
        },
        {
            "type": "Crystal Layer",
            "thickness": 1.5,
            "material": "Sapphire",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0
        },
        {
            "type": "Semi Infinite Isotropic Layer",
            "permittivity": 1.,
        }
    ]
})
    
    return payload


def mock_azimuthal_payload():
    payload = json.dumps({
        "ScenarioData": {
        "type": "Azimuthal",
        "azimuthalAngle": {"min": 0, "max": 360},
        "incidentAngle": 45,
    },
    "Layers": [
        {
            "type": "Ambient Incident Layer",
            "permittivity": 5.5
        },
        {
            "type": "Isotropic Middle-Stack Layer",
            "thickness": 1.
        },
        {
            "type": "Crystal Layer",
            "thickness": 0.1,
            "material": "Quartz",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 90,
            "rotationZType": "relative"
        },
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Quartz",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0,
            "rotationZType": "relative"
        }]
    })

    return payload


def mock_dispersion_payload():
    payload = json.dumps({
    "ScenarioData": {
        "type": "Dispersion",
        "azimuthalAngle": {"min": 0, "max": 360},
        "incidentAngle": {"min": 0, "max": 90},
        "frequency": 525
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
            "rotationZType": "relative"
        },
        {
            "type": "Semi Infinite Isotropic Layer",
            "permittivity": 1.
        }
    ],
    })

    return payload
