#!/bin/bash

func_config="function.yaml"
func_root=$(dirname "$func_config")

# docker build -t yolov8 .
docker build -t yolov8:latest.  
nuctl deploy --project-name llm --path "$func_root" --file "$func_config" --platform local