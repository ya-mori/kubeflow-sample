FROM python:3.9-slim-buster

RUN pip install --upgrade pip && pip install poetry==1.1.5

COPY ./data /root/data

COPY ./kubeflow-preprocessor /root/kubeflow-preprocessor
WORKDIR /root/kubeflow-preprocessor

RUN poetry export -f requirements.txt > requirements.txt && pip install -r requirements.txt

ENTRYPOINT ["python", "-m", "kubeflow_preprocessor"]
