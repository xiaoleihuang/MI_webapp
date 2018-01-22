#!/bin/bash
conda env create -f environment.yml
source activate myenv
pip install -r requirements.txt
conda install nomkl --yes
python -m nltk.downloader punkt stopwords wordnet
