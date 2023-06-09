import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from flask_compress import Compress

app = Flask(__name__)
CORS(app)
Compress(app)

from layers import Structure

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@app.route('/api/calculate', methods=['POST'])
def main():
    payload = request.json
    structure = Structure()
    structure.execute(payload)
    reflectivity = {
        "r_pp": {"real": structure.r_pp.numpy().real, "imag": structure.r_pp.numpy().imag},
        "r_ps": {"real": structure.r_ps.numpy().real, "imag": structure.r_ps.numpy().imag},
        "r_sp": {"real": structure.r_sp.numpy().real, "imag": structure.r_sp.numpy().imag},
        "r_ss": {"real": structure.r_ss.numpy().real, "imag": structure.r_ss.numpy().imag},
    }
    reflectivity = json.dumps(reflectivity, cls=NumpyEncoder)
    return jsonify(reflectivity)


app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
