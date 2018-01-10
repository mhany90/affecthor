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

# desired portions of the data (allowed: train, dev, test, traindev)
SETS=("train" "dev" "test" "traindev")

# affects considered in subtasks of type EI
AFFECTS=("anger" "fear" "joy" "sadness")

# sentiment lexicon directory
SENTLEXDIR=$DATADIR/lexicons

# CMU Tweet NLP directory
TWEETNLPDIR="$TOOLDIR/ark-tweet-nlp-0.3.2"

# weka directory
WEKADIR="$TOOLDIR/weka-3-9-1"


# download data and normalise naming
echo '[INFO] Downloading SemEval-2018 data...'
. $DATADIR/download_data.sh
echo '[INFO] Data has been downloaded'

# clean unvalid words from the data
echo '[INFO] Cleaning SemEval-2018 data...'
. $DATADIR/clean_data.sh
echo '[INFO] Data has been cleaned'

# dowload and format Arabic sentiment lexicons
echo '[INFO] Preparing Arabic/Spanish sentiment lexicons...'
. $DATADIR/download_lexicons.sh
echo '[INFO] Arabic/Spanish sentiment lexicons are ready'

