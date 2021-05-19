VERSION:=latest

build:
	docker build -t asia.gcr.io/${PROJECT}/kubeflow-sample/kubeflow-trainer:${VERSION} -f Dockerfile .
	docker push asia.gcr.io/${PROJECT}/kubeflow-sample/kubeflow-trainer:${VERSION}
