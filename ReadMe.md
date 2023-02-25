# Running the poi clustering api

## step 1

docker build -t entitymatching_api .

## step 2

docker run -d --name clustering-api -p 80:80 entitymatching_api3

### You can access the api as below in local

<http://localhost:8000/clusterId/{placeId>}

<http://localhost:8000/clusterId/1aa4a06b-ec67-45e5-beaf-ca464c3bea2c>

### Api can be accessed as below if running on aws server

<http://172.29.182.110/clusterId/353f339b-7322-4972-b97a-637d0e7f4220>

### docker notes

if you get below error:
Cannot connect to the Docker daemon at unix:///var/run/docker.sock.

run below command to start the docker service

sudo systemctl start docker

pip install "uvicorn[standard]" gunicorn
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 0