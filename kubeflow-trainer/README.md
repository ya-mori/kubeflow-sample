
```shell
# setup
poetry install

# run local machine
poetry run python -m kubeflow_trainer

# build docker image
make build PROJECT=`your gcp project id`
```