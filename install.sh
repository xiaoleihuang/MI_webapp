#!/bin/bash
conda create --name myenv --file myenv.txt
source activate myenv
pip install -r requirements.txt
conda install nomkl --yes
python -m nltk.downloader punkt stopwords wordnet
