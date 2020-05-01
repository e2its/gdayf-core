#!/bin/bash

myip="localhost"

docker-compose -f /home/e2its/docker/scripts/yaml/gdayf-platform.yml up -d unbuntu-mongodb 
docker-compose -f /home/e2its/docker/scripts/yaml/gdayf-platform.yml up -d ubuntu-h2o
docker-compose -f /home/e2its/docker/scripts/yaml/gdayf-platform.yml up -d namenode hive-metastore-postgresql
docker-compose -f /home/e2its/docker/scripts/yaml/gdayf-platform.yml up -d datanode hive-metastore
docker-compose -f /home/e2its/docker/scripts/yaml/gdayf-platform.yml up -d hive-server
docker-compose -f /home/e2its/docker/scripts/yaml/gdayf-platform.yml up -d spark-master spark-worker_1 spark-worker_2 hue

my_ip="localhost"
echo "mongodb: : htttp://${my_ip}:2717"
echo "H2O server: htttp://${my_ip}:54321"
echo "Namenode: http://${my_ip}:50070"
echo "Datanode: http://${my_ip}:50075"
echo "Spark-master: http://${my_ip}:8080"


