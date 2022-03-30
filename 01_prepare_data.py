import tensorflow as tf
from utils import create_runner, upload_dataset

keras = tf.keras


with create_runner("prepare-data") as run:
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    upload_dataset(run, "train", x_train, y_train)
    upload_dataset(run, "test", x_test, y_test)
