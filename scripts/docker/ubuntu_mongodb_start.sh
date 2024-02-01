#!/bin/bash -x
sudo docker run -it -d --rm --name mongodb --network=host -v /home/e2its/docker/volumes/mongodb/data:/data e2its/ubuntu-mongodb
