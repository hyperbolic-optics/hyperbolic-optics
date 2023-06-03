import math as m
import tensorflow as tf
from abc import ABC
from material_params import (Air, AmbientIncidentMedium, CalciteUpper, Quartz, Sapphire, Ambient_Exit_Medium)



class ScenarioSetup(ABC):
    def __init__(self, data):
        self.x_rotation = m.radians(float(data.get("rotationX")))
        self.y_rotation = m.radians(float(data.get("rotationY")))
        self.air_gap_thickness = float(data.get("airGapThickness")) * 1.e-4
        self.eps_prism = float(data.get("dielectricConstant"))
        self.material = self.create_material(data.get("material"))

    def create_material(self, material_name):
        material_classes = {"Quartz": Quartz, "Sapphire": Sapphire, "Calcite": CalciteUpper}
        return material_classes[material_name]()


class IncidentScenarioSetup(ScenarioSetup):
    def __init__(self, data):
        super().__init__(data)
        self.z_rotation = m.radians(float(data.get("azimuthalAngle")))
        incident_min = m.radians(float(data.get("incidentAngle").get("min")))
        incident_max = m.radians(float(data.get("incidentAngle").get("max")))
        self.incident_angle = tf.linspace(
            tf.constant(incident_min, dtype=tf.float32),
            tf.constant(incident_max, dtype=tf.float32),
            abs(int(m.degrees(incident_max - incident_min)//2))
            )


class AzimuthalScenarioSetup(ScenarioSetup):
    def __init__(self, data):
        super().__init__(data)
        self.incident_angle = m.radians(float(data.get("incidentAngle")))
        z_min = m.radians(float(data.get("azimuthalAngle").get("min")))
        z_max = m.radians(float(data.get("azimuthalAngle").get("max")))
        self.z_rotation = tf.linspace(
            tf.constant(z_min, dtype=tf.float32),
            tf.constant(z_max, dtype=tf.float32),
            abs(int(m.degrees(z_max - z_min)//2))
            )
    

class DispersionScenarioSetup(ScenarioSetup):
    def __init__(self, data):
        super().__init__(data)
        incident_min = m.radians(float(data.get("incidentAngle").get("min")))
        incident_max = m.radians(float(data.get("incidentAngle").get("max")))
        self.incident_angle = tf.linspace(
            tf.constant(incident_min, dtype=tf.float32),
            tf.constant(incident_max, dtype=tf.float32),
            abs(int(m.degrees(incident_max - incident_min)//2))
            )
        
        z_min = m.radians(float(data.get("azimuthalAngle").get("min")))
        z_max = m.radians(float(data.get("azimuthalAngle").get("max")))
        self.z_rotation = tf.linspace(
            tf.constant(z_min, dtype=tf.float32),
            tf.constant(z_max, dtype=tf.float32),
            abs(int(m.degrees(z_max - z_min)//2))
            )
        
        self.frequency = float(data.get("frequency"))
