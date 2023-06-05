import json



def mock_incident_payload():
    payload = json.dumps({
        "ScenarioData": {
        "type": "Incident",
        "incidentAngle": {"min": -90, "max": 90},
    },
    "Layers": [
        {
            "type": "prism",
            "eps_prism": 5.5
        },
        {
            "type": "air_gap",
            "thickness": 1.5
        },
        {
            "type": "crystal_layer",
            "thickness": 0.1,
            "material": "Calcite",
            "rotationX": 0,
            "rotationY": 45,
            "rotationZ": 0,
        },
        {
            "type": "semi_infinite_crystal",
            "material": "Quartz",
            "rotationX": 0,
            "rotationY": 0,
            "rotationZ": 20,
        }]
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
            "type": "prism",
            "eps_prism": 5.5
        },
        {
            "type": "air_gap",
            "thickness": 0.
        },
        {
            "type": "crystal_layer",
            "thickness": 1.,
            "material": "Sapphire",
            "rotationX": 0,
            "rotationY": 45,
            "rotationZ": 90,
            "rotationZType": "relative"
        },
        {
            "type": "semi_infinite_crystal",
            "material": "Quartz",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 45,
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
        "frequency": 496
    },
    "Layers": [
        {
            "type": "prism",
            "eps_prism": 5.5
        },
        {
            "type": "air_gap",
            "thickness": 1.5
        },
        {
            "type": "crystal_layer",
            "thickness": 1.,
            "material": "Sapphire",
            "rotationX": 0,
            "rotationY": 45,
            "rotationZ": 90,
            "rotationZType": "static"
        },
        {
            "type": "semi_infinite_crystal",
            "material": "Quartz",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0,
            "rotationZType": "relative"
        }
    ],
    })

    return payload
