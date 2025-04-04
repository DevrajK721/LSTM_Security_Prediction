import tensorflow as tf
import time

# Define matrix size (adjust if needed based on available memory)
matrix_size = 2000

# Create two random matrices
a = tf.random.normal((matrix_size, matrix_size))
b = tf.random.normal((matrix_size, matrix_size))

def measure_matmul_time(device, iterations=1000):
    with tf.device(device):
        # Warm up to ensure any lazy initialization is done
        c = tf.matmul(a, b)
        c.numpy()  # Force computation

        start_time = time.time()
        for _ in range(iterations):
            c = tf.matmul(a, b)
            c.numpy()  # Force the operation to complete
        end_time = time.time()
    return end_time - start_time

# Measure CPU execution time
cpu_time = measure_matmul_time('/CPU:0')
print("CPU time for 10 multiplications: {:.4f} seconds".format(cpu_time))

# Measure GPU execution time if available
if tf.config.list_physical_devices('GPU'):
    gpu_time = measure_matmul_time('/GPU:0')
    print("GPU time for 10 multiplications: {:.4f} seconds".format(gpu_time))
else:
    print("No GPU available.")