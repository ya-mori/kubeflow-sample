import os
from typing import Dict, Any

import kfp
from kfp.compiler import Compiler

from kubeflow_pipeline.pipeline_registory import get_pipeline


def compile_pipeline(name: str):
    Compiler().compile(
        pipeline_func=get_pipeline(name=name),
        package_path=f"../data/11_kubeflow_files/{name}.yaml"
    )


def execute_pipeline(name: str, params: Dict[str, Any]):
    compile_pipeline(name=name)
    client = kfp.Client(host=os.getenv("KF_HOST"))
    experiment = client.create_experiment(name=name)
    client.run_pipeline(
        experiment_id=experiment.id,
        job_name=name,
        pipeline_package_path=f"../data/11_kubeflow_files/{name}.yaml",
        params=params
    )
