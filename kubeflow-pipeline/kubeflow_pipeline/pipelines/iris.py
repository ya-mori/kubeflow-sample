from kfp import dsl
from kfp.dsl import InputArgumentPath, PipelineParam
from kubernetes.client import V1EnvVar


@dsl.component
def preprocess(
    project: str,
) -> dsl.ContainerOp:
    train_path = "/tmp/data/02_features/iris/train/dataset.tfrecord"
    test_path = "/tmp/data/02_features/iris/test/dataset.tfrecord"

    return dsl.ContainerOp(
        name="preprocess iris",
        image=f"asia.gcr.io/{project}/kubeflow-sample/kubeflow-preprocessor:latest",
        container_kwargs={
            "env": [
                V1EnvVar("TRAIN_TFRECORD_DIR_IRIS", train_path),
                V1EnvVar("TEST_TFRECORD_DIR_IRIS", test_path),
            ],
        },
        file_outputs={
            "train_path": train_path,
            "test_path": test_path,
        },
    )


@dsl.component
def train(
    project: str,
    train_path: PipelineParam,
    test_path: PipelineParam,
) -> dsl.ContainerOp:
    model_path = "/tmp/data/03_models/iris.h5"
    return dsl.ContainerOp(
        name="train iris",
        image=f"asia.gcr.io/{project}/kubeflow-sample/kubeflow-trainer:latest",
        container_kwargs={
            "env": [
                V1EnvVar("MODEL_DIR_IRIS", model_path),
            ],
        },
        arguments=[
            "--train_dir",
            InputArgumentPath(train_path),
            "--test_dir",
            InputArgumentPath(test_path),
        ],
    )


@dsl.pipeline(name="iris pipeline", description="iris data training pipeline")
def iris_pipeline(project):
    preprocessed = preprocess(project)
    preprocessed.execution_options.caching_strategy.max_cache_staleness = "P0D"
    trained = train(
        project, preprocessed.outputs["train_path"], preprocessed.outputs["test_path"]
    ).after(preprocessed)
    trained.execution_options.caching_strategy.max_cache_staleness = "P0D"
