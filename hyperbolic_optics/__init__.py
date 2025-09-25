"""
Hyperbolic Optics Simulation Package

4x4 Transfer Matrix Method for Anisotropic Multilayer Structures
"""

# Must set environment variables BEFORE any tensorflow imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages

# Now suppress warnings
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings for clean output

# Import tensorflow after setting environment variables
# Don't set any TensorFlow logging here since environment variables handle it
try:
    import tensorflow as tf
    # Disable TensorFlow warnings completely
    tf.get_logger().setLevel('ERROR')
except ImportError:
    pass  # TensorFlow not installed

__version__ = "0.1.8"