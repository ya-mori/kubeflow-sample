# kubeflow sample

Kubeflow pipelines sample using `kfp` 

## How to set up

```shell
export KF_HOST={your kubeflow host}

poetry install
```


## Run sample 

`kubeflow_pipeline/iris.py`
```shell
poetry run python -m kubeflow_pipeline --mode run --pipeline iris --params project=`your gcp project id`
# or
poetry run python -m kubeflow_pipeline --mode compile --pipeline iris --params project=`your gcp project id`
```

## Ref

https://www.kubeflow.org/docs/components/pipelines/sdk/sdk-overview/
