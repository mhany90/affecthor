#!/bin/bash
# this is a general setup script for this project


# resume training with option --stage N, -s N
stage=0

# install all tools again with option --install-tools, -t
newtools=0

# download data again with option --fetch-data, -d
newdata=0

# apply weka filters again with option --weka-filters, -f
newfilters=0


# set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands'
set -e
set -u
set -o pipefail
#set -x


# define directory locations

# root directory for this project
ROOTDIR=$PWD

# data directory
DATADIR=$PWD/data

# source directory
SRCDIR=$PWD/src

# tools directory
TOOLDIR=$PWD/tools

# models directory
MODELDIR=$PWD/models

# utils directory
UTILSDIR=$PWD/utils


# ensure script runs from the root directory
if ! [ -x "$PWD/setup.sh" ]; then
    echo '[INFO] You must run setup.sh from the root directory'
    exit 1
fi

# transform long options to short ones and parse them
for arg in "$@"; do
    shift
    case "$arg" in
        "--stage") set -- "$@" "-s" ;;
        "--fetch-data") set -- "$@" "-d" ;;
        "--install-tools") set -- "$@" "-t" ;;
        "--weka-filters") set -- "@" "-f" ;;
        *) set -- "$@" "$arg"
    esac
done

while getopts s:dt option
do
    case "${option}"
    in
        s) stage=${OPTARG};;
        d) newdata=1;;
        t) newtools=1;;
        f) newfilters=1;;
    esac
done


## STAGE 0 - set up required tools
if [ $stage -le 0 ]; then
    echo '[INFO] Setting up required tools...'
    . $TOOLDIR/prepare_tools.sh
    echo '[INFO] Finished setting up tools'
fi


## STAGE 1 - download and prepare data
if [ $stage -le 1 ]; then
    echo '[INFO] Preparing data...'
    . $DATADIR/prepare_data.sh
    echo '[INFO] Finished preparing data'
fi


# STAGE 2 - apply weka filters
if [ $stage -le 2 ]; then
    echo '[INFO] Applying feature filters...'
    . $DATADIR/filter_features.sh
    echo '[INFO] Finished feature filtering'
fi

