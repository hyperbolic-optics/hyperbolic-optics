# Basic Concepts

## Overview

The hyperbolic-optics package uses the **4×4 transfer matrix method** to calculate reflection and transmission coefficients for multilayer anisotropic structures.

## Key Components

### Structure


The `Structure` class is the main interface for setting up and running simulations. It:

- Defines the geometry (layers and materials)
- Sets up the scenario (angles, frequencies)
- Calculates reflection coefficients

### Layers

A structure consists of multiple layers:


1. **Ambient Incident Layer**: The incident medium (e.g., prism)
2. **Middle Layers**: Can be isotropic or anisotropic, finite thickness
3. **Exit Layer**: Usually semi-infinite, can be isotropic or anisotropic

### Materials

Materials are defined by their permittivity (ε) and permeability (μ) tensors:


- **Uniaxial**: Single optical axis (Quartz, Calcite, Sapphire)
- **Biaxial/Monoclinic**: Two or three optical axes (Gallium Oxide)
- **Isotropic**: No optical axis (Air, glass)
- **Arbitrary**: Custom-defined tensors

## Reflection Coefficients

The package calculates four reflection coefficients:


- **r_pp**: p-polarized → p-polarized
- **r_ss**: s-polarized → s-polarized  
- **r_ps**: p-polarized → s-polarized
- **r_sp**: s-polarized → p-polarized

Reflectivity is calculated as: $R = |r|^2$

## Coordinate System


- **x-axis**: Parallel to the interface, in the plane of incidence
- **y-axis**: Parallel to the interface, perpendicular to plane of incidence
- **z-axis**: Normal to the interface (propagation direction)

### Rotations

Materials can be rotated using Euler angles:


- **rotationY**: Rotation around y-axis (often the optical axis tilt)
- **rotationZ**: Rotation around z-axis (azimuthal rotation)

## Transfer Matrix Method

The 4×4 transfer matrix method tracks both electric and magnetic field components through each layer. For each layer:


1. Calculate the Berreman matrix (describes wave propagation)
2. Find eigenvalues and eigenvectors (wave modes)
3. Construct the transfer matrix
4. Multiply matrices for all layers
5. Extract reflection coefficients

## Mueller Matrices

Mueller matrices describe how polarization states transform upon reflection:

$$\mathbf{S}_{out} = \mathbf{M} \cdot \mathbf{S}_{in}$$

Where $\mathbf{S}$ is the Stokes vector: $[S_0, S_1, S_2, S_3]^T$

The package can:

1. Calculate Mueller matrices from reflection coefficients
2. Simulate optical components (polarizers, wave plates)
3. Calculate Stokes parameters and polarization properties

## Units


- **Frequency**: cm⁻¹ (wavenumbers)
- **Thickness**: mm (converted to cm internally)
- **Angles**: degrees (converted to radians internally)
- **Permittivity/Permeability**: dimensionless