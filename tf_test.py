import tensorflow as tf
import numpy as np

print(f"TensorFlow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Simple TensorFlow operation
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[1, 1], [1, 1]])
print("TensorFlow operation result:")
print(tf.matmul(a, b))

# Simple Keras model (without explicit input shape)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Generate some random data
x = np.random.random((1000, 5))
y = np.random.random((1000, 1))

# Compile and fit the model
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=1, verbose=1)

print("Test completed successfully!")