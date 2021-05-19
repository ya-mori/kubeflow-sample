# kubeflow sample

Kubeflow pipelines sample using `kfp` 

## Package structure

```
.
├── README.md
├── data
│   ├── 01_inputs
│   ├── 02_features
│   ├── 03_models
│   ├── 10_kubeflow_file_versions
│   └── 11_kubeflow_files
├── kubeflow-pipeline # Manage the ml pipeline by kfp
│   ├── README.md
│   ├── kubeflow_pipeline
│   ├── poetry.lock
│   ├── pyproject.toml
│   └── tests
├── kubeflow-preprocesser # Manage the ml prerpcessing by pandas
│   ├── README.rst
│   ├── kubeflow_preprocesser
│   ├── poetry.lock
│   ├── pyproject.toml
│   └── tests
├── kubeflow-trainer # Manage the ml training by pandas
│   ├── README.md
│   ├── kubeflow_trainer
│   ├── poetry.lock
│   ├── pyproject.toml
│   └── tests
├── notebooks
│   ├── ex_iris_1_pre.ipynb
│   └── ex_iris_1_train.ipynb
├── poetry.lock
└── pyproject.toml # Depends on kubeflow-preprocessor, kubeflow-trainer, kubeflow-pipeline

16 directories, 14 files
```

## Ref

https://www.kubeflow.org/docs/components/pipelines/sdk/sdk-overview/
https://gist.github.com/netj/8836201#file-iris-csv
