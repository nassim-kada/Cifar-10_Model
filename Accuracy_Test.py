import tensorflow as tf
from tensorflow import keras
(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_test = x_test / 255.0

model = keras.models.load_model('m_s1.h5')

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)

print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
