import tensorflow as tf
import cv2
import numpy as np
import wandb
from pathlib import Path

from utils import create_images, download_dataset, create_runner, download_model

keras = tf.keras

print("TensorFlow version:", tf.__version__)

model_dir = Path("model_files/model")
mnist = keras.datasets.mnist

with create_runner("eval") as run:
    x_test, y_test = download_dataset(run, "test")
    download_model(run, model_dir, version="latest")

    test_images = x_test[0:10]

    model = keras.models.load_model(model_dir)
    prediction = np.argmax(model.predict_on_batch(x_test), axis=1)

    m = keras.metrics.Accuracy()
    m.update_state(y_test, prediction)
    accuracy = round(m.result().numpy()*100, 2)
    print(f"Accuracy: {accuracy}%")

    images = create_images(x_test[0:20], prediction[:20])
    run.log({'accuracy': accuracy})
    run.log({"evaluation examples": wandb.Image(images, caption="evaluation examples")})
    cv2.imshow('image', images)
    cv2.waitKey(0)
