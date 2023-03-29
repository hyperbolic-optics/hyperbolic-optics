import numpy as np
import material_params
import plots
from berreman import layer_matrix_incidence, layer_matrix_incidence_azimuth
from anisotropy_utils import anisotropy_rotation


def compute_kx(eps_prism, incident_angle):
    return np.sqrt(eps_prism) * np.sin(incident_angle)


def reflection_coefficients(T):

    bottom_line = (T[...,0,0] * T[...,2,2] - T[...,0,2] * T[...,2,0])
    r_pp = (T[...,0,0] * T[...,3,2] - T[...,3,0] * T[...,0,2]) / bottom_line
    r_ps = (T[...,0,0] * T[...,1,2] - (T[...,1,0] * T[...,0,2])) / bottom_line
    r_sp = (T[...,3,0] * T[...,2,2] - T[...,3,2] * T[...,2,0]) / bottom_line
    r_ss = (T[...,1,0] * T[...,2,2] - T[...,1,2] * T[...,2,0]) / bottom_line  
    
    return np.array([r_pp, r_ps, r_sp, r_ss])


def main_quartz_theta():
    eps_prism = 5.5
    air_gap_thickness = 1.5e-4

    # Create quartz material and define frequency-related variables
    quartz = material_params.Quartz(300)
    frequency = quartz.frequency
    k0 = frequency * 2. * np.pi

    # Calculate incident angle and kx
    incident_angle = np.linspace(-np.pi/2., np.pi/2., quartz.frequency_length)
    kx = compute_kx(eps_prism, incident_angle)

    # Define rotation angles
    anisotropy_rotation_y = np.radians(90.)
    rotation_z = np.radians(45.)
    anisotropy_rotation_x = np.radians(0.)

    # Create permittivity tensors for quartz and air
    quartz_tensor = quartz.fetch_permittivity_tensor()
    non_magnetic_tensor = np.tile(material_params.Air().construct_tensor_singular(), (quartz.frequency_length, 1, 1))

    rotated_quartz_tensor = np.squeeze(anisotropy_rotation(quartz_tensor, anisotropy_rotation_y, rotation_z, anisotropy_rotation_x), axis=1)

    # Construct the layers
    prism_layer = material_params.Ambient_Incident_Prism(eps_prism, incident_angle).construct_tensor()
    air_layer = np.linalg.inv(layer_matrix_incidence(non_magnetic_tensor, non_magnetic_tensor, kx, k0, air_gap_thickness))
    quartz_layer = layer_matrix_incidence(rotated_quartz_tensor, non_magnetic_tensor, kx, 0, 0, quartz=True)

    # Calculate the transfer matrix and reflectivities
    transfer = prism_layer @ air_layer @ quartz_layer
    reflectivities = reflection_coefficients(transfer)

    # # Plot the reflectivities

    plot_type = 'theta'
    plots.contour_plot(plot_type, 
                       frequency, 
                       incident_angle, 
                       air_gap_thickness, 
                       None, 
                       anisotropy_rotation_x, 
                       anisotropy_rotation_y, 
                       rotation_z, 
                       reflectivities)

def main_quartz_rotation_z():
    eps_prism = 5.5
    air_gap_thickness = 0.e-4
    quartz = material_params.Quartz(300)
    frequency = quartz.frequency
    k0 = frequency * 2. * np.pi

    incident_angle = np.radians(45.) 
    kx = np.sqrt(eps_prism) * np.sin(incident_angle)

    # Define rotation angles
    anisotropy_rotation_y = np.radians(90.)
    rotation_z = np.linspace(np.radians(0), np.radians(360), 360) + np.pi/2.
    anisotropy_rotation_x = np.radians(0.)

    quartz_tensor = quartz.fetch_permittivity_tensor()
    non_magnetic_tensor = np.tile(material_params.Air().construct_tensor_singular(), (quartz.frequency_length, 1, 1))

    rotated_quartz_tensors = anisotropy_rotation(quartz_tensor, anisotropy_rotation_y, rotation_z, anisotropy_rotation_x)
    non_magnetic_tensor = anisotropy_rotation(non_magnetic_tensor, anisotropy_rotation_y, rotation_z, anisotropy_rotation_x)

    prism_layer = material_params.Ambient_Incident_Prism(eps_prism, incident_angle).construct_tensor_singular(eps_prism, incident_angle)
    
    air_layer = np.linalg.inv(layer_matrix_incidence_azimuth(non_magnetic_tensor, non_magnetic_tensor, kx, k0, air_gap_thickness))
    
    quartz_layer = layer_matrix_incidence_azimuth(rotated_quartz_tensors, non_magnetic_tensor, kx, 0, 0, quartz=True)

    transfer = prism_layer[np.newaxis, ...] @ air_layer @ quartz_layer

    reflectivities = reflection_coefficients(transfer)

    plot_type = 'azimuth'
    plots.contour_plot(plot_type, 
                       frequency, 
                       rotation_z, 
                       air_gap_thickness, 
                       incident_angle, 
                       anisotropy_rotation_x, 
                       anisotropy_rotation_y, 
                       None, 
                       reflectivities)

def main_quartz_rotation_y():
    eps_prism = 5.5
    air_gap_thickness = 0.e-4

    quartz = material_params.Quartz(300)
    frequency = quartz.frequency
    k0 = frequency * 2. * np.pi

    incident_angle = np.radians(45) 
    kx = np.sqrt(eps_prism) * np.sin(incident_angle)

    # Define rotation angles
    anisotropy_rotation_y = np.linspace(np.radians(0), np.radians(360), 300)
    rotation_z = np.radians(45.)
    anisotropy_rotation_x = np.radians(45.)

    quartz_tensor = quartz.fetch_permittivity_tensor()
    non_magnetic_tensor = np.tile(material_params.Air().construct_tensor_singular(), (quartz.frequency_length, 1, 1))

    rotated_quartz_tensors = anisotropy_rotation(quartz_tensor, anisotropy_rotation_y, rotation_z, anisotropy_rotation_x)
    non_magnetic_tensor = anisotropy_rotation(non_magnetic_tensor, anisotropy_rotation_y, rotation_z, anisotropy_rotation_x)

    prism_layer = material_params.Ambient_Incident_Prism(eps_prism, incident_angle).construct_tensor_singular(eps_prism, incident_angle)
    
    air_layer = np.linalg.inv(layer_matrix_incidence_azimuth(non_magnetic_tensor, non_magnetic_tensor, kx, k0, air_gap_thickness))
    
    quartz_layer = layer_matrix_incidence_azimuth(rotated_quartz_tensors, non_magnetic_tensor, kx, 0, 0, quartz=True)

    transfer = prism_layer[np.newaxis, ...] @ air_layer @ quartz_layer

    reflectivities = reflection_coefficients(transfer)

    plot_type = 'y_anisotropy'
    plots.contour_plot(plot_type, 
                       frequency, 
                       anisotropy_rotation_y, 
                       air_gap_thickness, 
                       incident_angle, 
                       anisotropy_rotation_x, 
                       None, 
                       rotation_z, 
                       reflectivities)

def main_quartz_rotation_x():
    eps_prism = 5.5
    air_gap_thickness = 0.e-4

    quartz = material_params.Quartz(300)
    frequency = quartz.frequency
    k0 = frequency * 2. * np.pi

    incident_angle = np.radians(45) 
    kx = np.sqrt(eps_prism) * np.sin(incident_angle)

    # Define rotation angles
    anisotropy_rotation_y = np.radians(45.)
    rotation_z = np.radians(45.)
    anisotropy_rotation_x = np.linspace(np.radians(0), np.radians(360), 300)

    quartz_tensor = quartz.fetch_permittivity_tensor()
    non_magnetic_tensor = np.tile(material_params.Air().construct_tensor_singular(), (quartz.frequency_length, 1, 1))

    rotated_quartz_tensors = anisotropy_rotation(quartz_tensor, anisotropy_rotation_y, rotation_z, anisotropy_rotation_x)
    non_magnetic_tensor = anisotropy_rotation(non_magnetic_tensor, anisotropy_rotation_y, rotation_z, anisotropy_rotation_x)

    prism_layer = material_params.Ambient_Incident_Prism(eps_prism, incident_angle).construct_tensor_singular(eps_prism, incident_angle)
    
    air_layer = np.linalg.inv(layer_matrix_incidence_azimuth(non_magnetic_tensor, non_magnetic_tensor, kx, k0, air_gap_thickness))
    
    quartz_layer = layer_matrix_incidence_azimuth(rotated_quartz_tensors, non_magnetic_tensor, kx, 0, 0, quartz=True)

    transfer = prism_layer[np.newaxis, ...] @ air_layer @ quartz_layer

    reflectivities = reflection_coefficients(transfer)

    plot_type = 'x_anisotropy'
    plots.contour_plot(plot_type, 
                       frequency, 
                       anisotropy_rotation_x, 
                       air_gap_thickness, 
                       incident_angle, 
                       None, 
                       anisotropy_rotation_y, 
                       rotation_z, 
                       reflectivities)


if __name__ == "__main__":
    # main_quartz_rotation_x()
    # main_quartz_rotation_y()
    main_quartz_rotation_z()
    # main_quartz_theta()
    