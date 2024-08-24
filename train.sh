#!/bin/bash

pip install -r requirements.txt

cd GPT

echo "Starting to Train"
python3 hyperparameter_search.py
