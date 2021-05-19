VERSION:=latest

build:
	docker build -t asia.gcr.io/${PROJECT}/kubeflow-sample/kubeflow-preprocessor:${VERSION} -f Dockerfile ../
	docker push asia.gcr.io/${PROJECT}/kubeflow-sample/kubeflow-preprocessor:${VERSION}

run/local:
	poetry run python -m kubeflow_preprocessor
