#!/bin/bash

# load UCI data 
wget -P data/UCI/ https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip data/UCI/UCI\ HAR\ Dataset.zip -d data/UCI/
rm data/UCI/UCI\ HAR\ Dataset.zip
