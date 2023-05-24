import math as m
import base64
import tensorflow as tf
import functools
import operator
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=['http://localhost:8080'])

from anisotropy_utils import anisotropy_rotation_one_value
from berreman import transfer_matrix_wrapper, reflection_coefficients
from device_config import run_on_device
from material_params import (Air, Ambient_Incident_Prism, CalciteUpper, Quartz, Sapphire)
from plots import contour_plot_simple_incidence


def mock_payload():
    payload = json.dumps({
        "scenario":"Incident",
        "rotationX":"0",
        "rotationY":"45",
        "airGapThickness":"1.5",
        "dielectricConstant":5.5,
        "material":"Quartz",
        "range":{"min":-90,"max":90},
        "azimuthalAngle":"0"
        })
    
    return payload


def parse_json_initial(data):

    # Extract parameters from JSON
    scenario = data.get("scenario")
    x_rotation = m.radians(float(data.get("rotationX")))
    y_rotation = m.radians(float(data.get("rotationY")))
    z_rotation = m.radians(float(data.get("azimuthalAngle")))
    air_gap_thickness = float(data.get("airGapThickness")) * 1.e-4
    eps_prism = float(data.get("dielectricConstant"))
    material_name = data.get("material")

    if material_name == 'Quartz':
        material = Quartz(frequency_length=300, run_on_device_decorator=run_on_device)
    elif material_name == 'Sapphire':
        material = Sapphire(frequency_length=300, run_on_device_decorator=run_on_device)
    elif material_name == 'Calcite':
        material = CalciteUpper(frequency_length=300, run_on_device_decorator=run_on_device)
    else:
        raise ValueError("Material not found")

    if scenario == 'Incident':
        z_rotation = m.radians(float(data.get("azimuthalAngle")))
        incident_min = m.radians(float(data.get("range").get("min")))
        incident_max = m.radians(float(data.get("range").get("max")))

        return {
            "eps_prism": eps_prism,
            "air_gap_thickness": air_gap_thickness,
            "material": material,
            "x_rotation": x_rotation,
            "y_rotation": y_rotation,
            "z_rotation": z_rotation,
            "incident_min": incident_min,
            "incident_max": incident_max
        }
    
    elif scenario == 'Azimuthal':
        incident_angle = m.radians(float(data.get("incidentAngle")))
        z_min = m.radians(float(data.get("range").get("min")))
        z_max = m.radians(float(data.get("range").get("max")))

        return {
            "eps_prism": eps_prism,
            "air_gap_thickness": air_gap_thickness,
            "material": material,
            "x_rotation": x_rotation,
            "y_rotation": y_rotation,
            "incident_angle": incident_angle,
            "z_min": z_min,
            "z_max": z_max
        }


@run_on_device
def incident_api_situation(params):
    eps_prism = params.get("eps_prism")
    air_gap_thickness = params.get("air_gap_thickness")
    material = params.get("material")
    x_rotation = params.get("x_rotation")
    y_rotation = params.get("y_rotation")
    z_rotation = params.get("z_rotation")
    incident_min = params.get("incident_min")
    incident_max = params.get("incident_max")

    incident_angle = tf.linspace(
        tf.constant(incident_min, dtype=tf.float32),
        tf.constant(incident_max, dtype=tf.float32),
        abs(int(m.degrees(incident_max - incident_min)//2)),
    )

    k_x = tf.cast(tf.sqrt(eps_prism) * tf.sin(incident_angle), dtype=tf.complex64)
    k_0 = material.frequency * 2.0 * m.pi


    eps_tensor = material.fetch_permittivity_tensor()
    eps_tensor = anisotropy_rotation_one_value(
        eps_tensor, x_rotation, y_rotation, z_rotation
    )

    # Construct the non-magnetic tensor.
    non_magnetic_tensor = Air(
        run_on_device_decorator=run_on_device
    ).construct_tensor_singular()

    # Construct the air layer.
    air_layer = (
            transfer_matrix_wrapper(
                k_x,
                non_magnetic_tensor,
                non_magnetic_tensor,
                k_0,
                thickness=air_gap_thickness,
                mode = "airgap"
            )
        )


    # Construct the prism layer.
    incident_prism = Ambient_Incident_Prism(
        eps_prism, incident_angle, run_on_device_decorator=run_on_device
    )
    prism_layer = incident_prism.construct_tensor()

    #semi_infinite end layer
    semi_infinite_layer = transfer_matrix_wrapper(
        k_x,
        eps_tensor,
        non_magnetic_tensor,
        semi_infinite=True,
        mode = "single_rotation"
    )

    ### Reshaping
    prism_layer = prism_layer[tf.newaxis, ...]

    layers = [prism_layer, air_layer, semi_infinite_layer]

    # Apply matrix multiplication successively using functools.reduce and operator.matmul
    transfer_matrix = functools.reduce(operator.matmul, layers)

    ### Reflection Coefficient
    reflectivity_values = reflection_coefficients(transfer_matrix)

    reflectivity_values = tf.cast(reflectivity_values * tf.math.conj(reflectivity_values), dtype = tf.float32)

    return reflectivity_values.numpy(), material.frequency.numpy()

@app.route('/api/calculate', methods=['POST'])
def perform_calculation():
    payload = request.json
    params = parse_json_initial(payload)
    reflectivity, frequency = incident_api_situation(params)
    reflectivity_b64 = base64.b64encode(reflectivity.tobytes()).decode('utf-8')
    frequency_b64 = base64.b64encode(frequency.tobytes()).decode('utf-8')
    result_to_send = {
        'reflectivity': reflectivity_b64,
        'frequency': frequency_b64,
        'shape': reflectivity.shape
    }
    return jsonify(result_to_send)


if __name__ == '__main__':
    app.run(debug=True)
