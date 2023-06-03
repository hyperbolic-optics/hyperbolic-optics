import json



def mock_incident_payload():
    payload = json.dumps({
        "scenario":"Incident",
        "rotationX":0,
        "rotationY":45,
        "airGapThickness":0,
        "dielectricConstant":5.5,
        "material":"Quartz",
        "incidentAngle":{"min":-90,"max":90},
        "azimuthalAngle":0
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
        "rotationY": 90,
        "airGapThickness": 0.,
        "dielectricConstant": 5.5,
        "material":"Quartz",
        "azimuthalAngle": {"min":0, "max":360},
        "incidentAngle": {"min":0,"max":90},
        "frequency": 468
        })
    
    return payload

def mock_dispersion_payload_full():
    payload = json.dumps({
    "scenario": "Dispersion",
    "layers": [
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
            "thickness": 0.1,
            "material": "Quartz",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0
        },
        {
            "type": "semi_infinite_crystal",
            "material": "Quartz",
            "rotationX": 0,
            "rotationY": 0,
            "rotationZ": 0
        }
    ],
    "azimuthalAngle": {"min": 0, "max": 360},
    "incidentAngle": {"min": 0, "max": 90},
    "frequency": 468
    })
