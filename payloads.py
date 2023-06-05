import json



def mock_incident_payload():
    payload = json.dumps({
        "scenario":"Incident",
        "rotationX":0,
        "rotationY":90,
        "airGapThickness":0,
        "dielectricConstant":11.56,
        "material":"Quartz",
        "incidentAngle":{"min":-90,"max":90},
        "azimuthalAngle":45
        })
    
    return payload


def mock_azimuthal_payload():
    payload = json.dumps({
        "scenario":"Azimuthal",
        "rotationX":"0",
        "rotationY": 90,
        "airGapThickness":1.5,
        "dielectricConstant": 5.5,
        "material":"Quartz",
        "azimuthalAngle":{"min":0,"max":360},
        "incidentAngle":"45"
        })
    
    return payload


def mock_dispersion_payload():
    payload = json.dumps({
        "scenario":"Dispersion",
        "rotationX": 0,
        "rotationY": 45,
        "airGapThickness": 0.,
        "dielectricConstant": 5.5,
        "material":"Sapphire",
        "azimuthalAngle": {"min":0, "max":360},
        "incidentAngle": {"min":0,"max":90},
        "frequency": 525
        })
    
    return payload


def mock_dispersion_payload_full():
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
            "rotationZShift": 90,
            "rotationZType": "static"
        },
        {
            "type": "semi_infinite_crystal",
            "material": "Quartz",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZShift": 0,
            "rotationZType": "relative"
        }
    ],
    })

    return payload
