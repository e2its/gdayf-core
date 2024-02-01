#!/bin/bash

myip="localhost"

docker-compose -f ./yaml/gdayf-platform.yml up -d unbuntu-mongodb 
docker-compose -f ./yaml/gdayf-platform.yml up -d ubuntu-h2o-1 
docker-compose -f ./yaml/gdayf-platform.yml up -d namenode hive-metastore-postgresql
docker-compose -f ./yaml/gdayf-platform.yml up -d datanode hive-metastore
docker-compose -f ./yaml/gdayf-platform.yml up -d hive-server
docker-compose -f ./yaml/gdayf-platform.yml up -d spark-master spark-worker-1 hue
docker-compose -f ./yaml/gdayf-platform.yml up -d ubuntu-h2o-2 spark-worker-2 
docker-compose -f ./yaml/gdayf-platform.yml up -d ubuntu-h2o-3 spark-worker-3 


my_ip="localhost"
echo "mongodb: : htttp://${my_ip}:2717"
echo "H2O server: htttp://${my_ip}:54321"
echo "Namenode: http://${my_ip}:50070"
echo "Datanode: http://${my_ip}:50075"
echo "Spark-master: http://${my_ip}:8080"


