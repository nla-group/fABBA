docker build -f docker/Dockerfile -t fabba-jupyter .
docker run --rm -p 8888:8888 fabba-jupyter
