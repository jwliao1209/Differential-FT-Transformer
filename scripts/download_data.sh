#!/bin/bash

if [ ! -d "data" ]; then
    wget https://www.dropbox.com/s/o53umyg6mn3zhxy/data.tar.gz?dl=1 -O revisiting_models_data.tar.gz
    tar -xvf revisiting_models_data.tar.gz
fi
