import json
from structure import Structure
from payloads import mock_incident_payload, mock_azimuthal_payload, mock_dispersion_payload


def main():
    """
    Main function
    """
    payload = json.loads(mock_dispersion_payload())
    structure = Structure()
    structure.execute(payload)
    structure.plot()


if __name__ == '__main__':
    main()
