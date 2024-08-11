import sys
import tensorflow as tf

print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")

# List available devices
print("Available devices:")
print(tf.config.list_physical_devices())

# Simple TensorFlow operation
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[1, 1], [1, 1]])
c = tf.matmul(a, b)

print("TensorFlow operation result:")
print(c)

# Test GPU availability
print("Is built with CUDA:", tf.test.is_built_with_cuda())
print("Is GPU available:", tf.test.is_gpu_available())