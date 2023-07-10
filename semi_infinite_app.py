from structure import Structure
import json



def updating_payload(scenario, eps_prism, air_gap_thickness, rotationY, rotationZ):
    payload = json.dumps({
        "ScenarioData": {
        "type": scenario,
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


def main():
    payload = json.loads(updating_payload("Incident", 5.5, 0, 0, 0))
    structure = Structure()
    structure.execute(payload)

    response = {
        "payload": payload,
        "r_pp": structure.r_pp,
        "r_ps": structure.r_ps,
        "r_sp": structure.r_sp,
        "r_ss": structure.r_ss,
    }

    print(response.keys())

main()