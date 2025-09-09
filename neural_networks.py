import tensorflow as tf
import numpy as np

# 1. Prepare the data
x_data = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
y_data = np.array([[4.0], [7.0], [10.0], [13.0]], dtype=np.float32) # y = 3*x + 1

# 2. Define the model more succinctly using the Keras API
# Keras is the high-level API that sits on top of TensorFlow.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]) # A single neuron: y = w*x + b
])

# 3. Compile the model: Choose the optimizer and loss function
model.compile(optimizer='sgd', # Stochastic Gradient Descent
              loss='mean_squared_error')

# 4. Train the model (The training loop is abstracted away!)
print("Training the model...")
history = model.fit(x_data, y_data, epochs=1000, verbose=1) # verbose=0 hides the output

# 5. Test the trained model
print("\nLet's test the model!")
test_x = np.array([[5.0]], dtype=np.float32)
predicted_y = model.predict(test_x)
print(f'Input: 5 | Predicted output: {predicted_y[0][0]}') # Expect ~16

# Let's see the learned parameters
w, b = model.layers[0].get_weights()
print(f'Learned equation: y = {w[0][0]:.2f} * x + {b[0]:.2f}') # Should be ~3 and ~1