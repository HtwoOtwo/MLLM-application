#!/bin/bash

func_config="function.yaml"
func_root=$(dirname "$func_config")

docker build -t intervl:v1 .
docker run -d -p 8070:8070 -v /var/run/docker.sock:/var/run/docker.sock --name nuclio-dashboard quay.io/nuclio/dashboard:stable-amd64 
nuctl create project llm
nuctl deploy --project-name llm \
    --path "$func_root" \
    --file "$func_config" \
    --platform local \