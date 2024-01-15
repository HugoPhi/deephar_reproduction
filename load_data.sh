#!/bin/bash

# load UCI data 
if [ ! -d "data/UCI" ]; then
    echo -n "do you want to download UCI data? (y/n)"
    read answer

    answer_lowercase=$(echo "$answer" | tr '[:upper:]' '[:lower:]')

    if [ "$answer_lowercase" == "y" ]; then
        echo "downloading UCI data..."
        wget -P data/UCI/ https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
        unzip data/UCI/UCI\ HAR\ Dataset.zip -d data/UCI/
        rm data/UCI/UCI\ HAR\ Dataset.zip
    else
        echo "abort."
    fi
fi
