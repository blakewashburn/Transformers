import tensorflow as tf

print(tf.__version__)
# print(f'GPUs Available: {tf.test.is_gpu_available()}')
print(f'{tf.config.list_physical_devices("GPU")}')


