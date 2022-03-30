import shutil

import cv2
import numpy as np
import wandb
import tempfile
import pickle
import zipfile
from typeguard import typechecked
from typing import Optional, Tuple, Any
from pathlib import Path

_PROJECT_NAME = "mnist_cnn"
_WANDB_ENTITY = "netzeband"


def create_images(images, labels):
    single_image_height = images.shape[1]
    single_image_width = images.shape[2]
    number_of_images = images.shape[0]
    height = single_image_height+4+30
    width = (2+single_image_width)*number_of_images+2
    image = np.ones((height, width, 1), np.uint8)*255
    for i, test_image in enumerate(np.squeeze(images[:, ...])):
        x = 2+i*(single_image_width+2)
        y = 2
        image[y:y+single_image_height, x:x+single_image_width, 0] = test_image[:, :]*255
        cv2.putText(
            image, str(labels[i]),
            (x+5, y+single_image_height+single_image_height-2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    return image


@typechecked
def create_runner(description: str, config: Optional[dict] = None) -> wandb.sdk.wandb_run.Run:
    kwargs = {}
    if config is not None:
        kwargs['config']=config
    return wandb.init(project=_PROJECT_NAME, entity=_WANDB_ENTITY, job_type=description, **kwargs)


@typechecked
def upload_dataset(run: wandb.sdk.wandb_run.Run, description: str, x, y):
    with tempfile.TemporaryDirectory() as cache_dir:
        data_path = Path(cache_dir) / "training_data.pickle"
        with data_path.open("wb") as f:
            pickle.dump(dict(x=x, y=y), f)
        training_data_artifact = wandb.Artifact(description, type='dataset')
        training_data_artifact.add_file(str(data_path))
        run.log_artifact(training_data_artifact)


@typechecked
def upload_model(run: wandb.sdk.wandb_run.Run, model_path: Path, epoch: Optional[int] = None, name: str = "model"):
    with tempfile.TemporaryDirectory() as cache_dir:
        model_file_name = "model"
        if epoch is not None:
            model_file_name += '_' + str(epoch)
        model_file_name += ".zip"

        cache_dir = Path(cache_dir).expanduser().resolve(strict=True)
        model_file_path = cache_dir / model_file_name
        with zipfile.ZipFile(model_file_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in model_path.rglob('*'):
                zf.write(file, file.relative_to(model_path))
        assert model_file_path.exists()

        model_artifact = wandb.Artifact(name, type='model')
        model_artifact.add_file(str(model_file_path))
        run.log_artifact(model_artifact)


@typechecked
def download_dataset(run: wandb.sdk.wandb_run.Run, description: str, version: str = "latest") -> Tuple[Any, Any]:
    artifact_name = description + ':' + version
    artifact = run.use_artifact(artifact_name, type='dataset')
    with tempfile.TemporaryDirectory() as cache_dir:
        data_path = Path(cache_dir) / "training_data.pickle"
        artifact.download(cache_dir)
        with data_path.open("rb") as f:
            data = pickle.load(f)
            return data['x'], data['y']


@typechecked
def download_model(run: wandb.sdk.wandb_run.Run, model_path: Path, version: str = "latest", name: str = "model"):
    artifact_name = name + ':' + version
    artifact = run.use_artifact(artifact_name, type='model')
    with tempfile.TemporaryDirectory() as cache_dir:
        cache_dir = Path(cache_dir)
        artifact.download(str(cache_dir))
        model_files = list(cache_dir.glob('*.zip'))
        assert len(model_files) == 1, f"Cannot find a single model file in '{cache_dir}': {model_files}"
        model_file_path = model_files[0]

        if model_path.exists():
            print(f"clean up model path: {model_path}")
            shutil.rmtree(model_path)

        model_path.mkdir(exist_ok=True)
        with zipfile.ZipFile(model_file_path, 'r') as zip:
            zip.extractall(str(model_path))
        print(f"extracted model to path: {model_path}")
