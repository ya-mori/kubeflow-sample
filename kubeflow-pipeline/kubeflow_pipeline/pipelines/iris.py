from typing import Any

from kfp import dsl
from kfp.dsl import InputArgumentPath
from kubernetes.client import V1EnvVar


def preprocess(
        project: str,
):
    train_path = "/tmp/data/02_features/iris/train.tfrecord"
    test_path = "/tmp/data/02_features/iris/test.tfrecord"

    return dsl.ContainerOp(
        name='preprocess iris',
        image=f'asia.gcr.io/{project}/kubeflow-sample/kubeflow-preprocessor:latest',
        container_kwargs={
            "env": [
                V1EnvVar("TRAIN_TFRECORD_DIR_IRIS", train_path),
                V1EnvVar("TEST_TFRECORD_DIR_IRIS", test_path),
            ],
        },
        file_outputs={
            "train": train_path,
            "test": test_path,
        },
    )


def train(
        project: str,
        train_path: Any,
        test_path: Any,
):
    # train_path = "/tmp/data/02_features/iris/train.tfrecord"
    # test_path = "/tmp/data/02_features/iris/test.tfrecord"
    train_path_arg = InputArgumentPath(train_path)
    test_path_arg = InputArgumentPath(test_path)
    model_path = "/tmp/data/03_models/iris.h5"
    return dsl.ContainerOp(
        name='train iris',
        image=f'asia.gcr.io/{project}/kubeflow-sample/kubeflow-trainer:latest',
        container_kwargs={
            "env": [
                V1EnvVar("TRAIN_TFRECORD_DIR_IRIS", train_path_arg.path),
                V1EnvVar("TEST_TFRECORD_DIR_IRIS", test_path_arg.path),
                V1EnvVar("MODEL_DIR_IRIS", model_path),
            ],
        },
        artifact_argument_paths=[
            # InputArgumentPath(train_path),
            # InputArgumentPath(test_path),
            train_path_arg,
            test_path_arg,
        ],
        file_outputs={
            # "model": model_path,
        }
    )


@dsl.pipeline(
    name='iris pipeline',
    description='iris data training pipeline'
)
def iris_pipeline(project):
    preprocessed = preprocess(project)
    train(project, preprocessed.outputs["train"], preprocessed.outputs["test"])
