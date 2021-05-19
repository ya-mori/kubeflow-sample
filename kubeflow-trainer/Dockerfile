FROM python:3.9-slim-buster

RUN pip install --upgrade pip && pip install poetry==1.1.5

COPY . /root
WORKDIR /root

RUN poetry export -f requirements.txt > requirements.txt && pip install -r requirements.txt

ENTRYPOINT ["python", "-m", "kubeflow_trainer"]
