# Running the poi clustering api

## step 1

docker build -t entitymatching_api .

## step 2

docker run -d --name clustering-api -p 80:80 entitymatching_api3

### You can access the api as below in local

<http://localhost:8000/clusterId/{placeId>}
http://127.0.0.1:8000/match_score?poi1={placeId1>}&poi2={placeId2>}

http://127.0.0.1:8000/match_score?poi1=4ffce9cf-de12-427b-bb30-09ec95b51ccc&poi2=378cefe4-b784-4def-a803-0f4994e0d879

### docker notes

if you get below error:
Cannot connect to the Docker daemon at unix:///var/run/docker.sock.

run below command to start the docker service

sudo systemctl start docker

pip install "uvicorn[standard]" gunicorn
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 0
