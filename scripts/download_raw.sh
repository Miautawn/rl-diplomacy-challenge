#!/bin/sh

# change the directory to this repo
cd "$(dirname "$0")"
cd .. 

#run the dvc to download the zip file
dvc pull

#unzip the downloaded zip containing all the files
unzip data/raw_data/diplomacy-v1-27k-msgs.zip -d data/raw_data/

#remove the zip
rm data/raw_data/diplomacy-v1-27k-msgs.zip 

