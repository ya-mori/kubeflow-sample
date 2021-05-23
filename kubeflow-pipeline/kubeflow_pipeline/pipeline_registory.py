from typing import Callable

from kubeflow_pipeline.pipelines.iris import iris_pipeline

pipelines = dict(
    iris=iris_pipeline,
)


def get_pipeline(name: str) -> Callable:
    pipeline = pipelines.get(name)
    if pipeline is not None:
        return pipeline
    else:
        raise ValueError(f"no such pipeline name: {name}")
