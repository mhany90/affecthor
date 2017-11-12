#!/bin/bash
# this script fetches and converts all the data needed to train our models


# root directory where the datasets are located
URL_PREFIX="saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA"

# name of each subtask (EI: Emotion Intensity, V: Valence)
TASKS=("EI-reg" "EI-oc" "V-reg" "V-oc")

# type of classification for each subtask (reg: regression, oc: ordinal)
TYPES=("reg" "oc" "reg" "oc")

# possible languages
LANGS=("English" "Arabic" "Spanish")

# abbreviation for each language
CODES=("En" "Ar" "Es")

# desired portions of the data (allowed: train, dev, test)
SETS=("train" "dev" "test")

# affects considered in subtasks of type EI
AFFECTS=("anger" "fear" "joy" "sadness")


# download data and normalise naming
echo '[INFO] Downloading data...'
. $DATADIR/download_data.sh
echo '[INFO] Data has been downloaded'

# clean unvalid words from the data
echo '[INFO] Cleaning data...'
. $DATADIR/clean_data.sh
echo '[INFO] Data has been cleaned'

