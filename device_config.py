import tensorflow as tf

# Set the default device to GPU if available, otherwise use CPU
default_device = (
    "/device:CPU:0" if tf.config.list_physical_devices("GPU") else "/device:CPU:0"
)


def run_on_device(func):
    def wrapper(*args, **kwargs):
        with tf.device(default_device):
            return func(*args, **kwargs)

    return wrapper
