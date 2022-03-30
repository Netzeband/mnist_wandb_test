import shutil
import wandb
import tensorflow as tf
import datetime
import cv2
import numpy as np
from pathlib import Path
import logging
from wandb.keras import WandbCallback

from model import Model
from utils import create_images, create_runner, download_dataset, upload_model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)
log_dir = Path("logs/fit") / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_dir = Path("model_files/model")
model_dir.parent.mkdir(exist_ok=True)

parameter = {
    'batch_size': 32,
    'epochs': 5,
    'optimizer': {
        'learning_rate': 0.001,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-07,
        'amsgrad': False,
    },
    'model': {
    }
}

keras = tf.keras

print("TensorFlow version:", tf.__version__)

with create_runner("train", config=parameter) as run:
    x_train, y_train = download_dataset(run, "train")
    x_test, y_test = download_dataset(run, "test")

    test_images = x_test[0:10]

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=parameter['optimizer']['learning_rate'],
        beta_1=parameter['optimizer']['beta_1'],
        beta_2=parameter['optimizer']['beta_2'],
        epsilon=parameter['optimizer']['epsilon'],
        amsgrad=parameter['optimizer']['amsgrad'],
        name='Adam',
    )

    model = Model()
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    store_model_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_dir,
        save_weights_only=False,
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
    )

    def show_test_images(epoch, *args, **kwargs):
        test_labels = np.argmax(model.predict(test_images), axis=1)
        image = create_images(test_images, test_labels)
        wandb.log({"test examples": wandb.Image(image, caption="test examples")})
        cv2.imshow('image', image)
        cv2.waitKey(1)

    image_callback = keras.callbacks.LambdaCallback(on_train_begin=show_test_images, on_epoch_end=show_test_images)

    model.fit(
        x=x_train,
        y=y_train,
        epochs=parameter['epochs'],
        batch_size=parameter['batch_size'],
        validation_data=(x_test, y_test),
        callbacks=[tensorboard_callback, store_model_callback, image_callback, WandbCallback(save_model=False)]
    )

    upload_model(run, model_dir, epoch=parameter['epochs'])
