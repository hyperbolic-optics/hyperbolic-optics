"""
script.py

Takes JSON payload containing data on multilayer structure,
creates the structure object then produces the reflectivity spectra.
"""

import json
from structure import Structure
from payloads import mock_incident_payload, mock_azimuthal_payload, mock_dispersion_payload


def main():
    """
    Main function
    """
    mode = 'azimuthal'
    if mode == 'incident':
        payload = json.loads(mock_incident_payload())
    elif mode == 'azimuthal':
        payload = json.loads(mock_azimuthal_payload())
    elif mode == 'dispersion':
        payload = json.loads(mock_dispersion_payload())
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")
    structure = Structure()
    structure.execute(payload)
    structure.plot()


if __name__ == '__main__':
    main()
