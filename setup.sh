#!/bin/bash
# this is a general setup script for this project


# resume training with option --stage N
stage=0

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
        *) set -- "$@" "$arg"
    esac
done

while getopts s:dt option
do
    case "${option}"
    in
        s) stage=${OPTARG};;
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

