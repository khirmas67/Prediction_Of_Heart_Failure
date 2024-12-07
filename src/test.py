import os
import tempfile
import tensorflow as tf

# Get the Documents folder path (works for both Windows and macOS/Linux)
documents_dir = os.path.join(os.path.expanduser("~"), "Documents")

# Define a subfolder name for temporary files inside Documents
custom_temp_dir = os.path.join(documents_dir, "MyTempDir")

# Create the custom temporary directory if it doesn't exist
os.makedirs(custom_temp_dir, exist_ok=True)

# Set the temporary directory to the custom directory
tempfile.tempdir = custom_temp_dir  # Set the default temp directory for Python
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Optional: Suppress TensorFlow logs

# Confirm the custom temp directory
print(f"Custom temporary directory set to: {custom_temp_dir}")

# Example TensorFlow operation to confirm temp directory usage
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

print("TensorFlow is using the custom temporary directory.")
