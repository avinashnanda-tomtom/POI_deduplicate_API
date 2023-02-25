# FROM ubuntu:18.04
FROM tiangolo/uvicorn-gunicorn:python3.8

# Install base utilities
RUN apt-get update
RUN apt-get install -y htop vim
RUN apt-get install build-essential -y

WORKDIR /code
COPY ./app /code/app

COPY ./requirements.txt .


RUN pip install -r requirements.txt

RUN TOKENIZERS_PARALLELISM=false
ENV PYTHONPATH=$PYTHONPATH:/code
ENV PYTHONPATH=$PYTHONPATH:/code/app


WORKDIR /code   

COPY ./app /code/app 

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
CMD ["gunicorn", "app.main:app", "--workers", "4","--worker-class","uvicorn.workers.UvicornWorker","--bind", "0.0.0.0:80", "--timeout", "0"]