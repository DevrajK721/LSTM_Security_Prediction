import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# List available GPU devices
gpus = tf.config.list_physical_devices('GPU')
print("Number of GPUs available:", len(gpus))

# Detailed check
if gpus:
    print("GPU is available!")
    # Optionally, print GPU details
    for gpu in gpus:
        print("GPU details:", gpu)
else:
    print("GPU is NOT available.")