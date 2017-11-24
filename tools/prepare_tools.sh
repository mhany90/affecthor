#!/bin/bash
# this script downloads and installs external tools


TWEETNLPDIR="$TOOLDIR/ark-tweet-nlp-0.3.2"

EMOINTDIR="$TOOLDIR/EmoInt"

WEKADIR="$TOOLDIR/weka-3-9-1"


# CMU Tweet NLP
echo '[INFO] Preparing CMU Tweet NLP...'
if [ ! -d "$TWEETNLPDIR" ] || [ $newtools -ge 1 ]; then
    rm -rf $TWEETNLPDIR
    pushd $TOOLDIR > /dev/null
    wget -q "https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/ark-tweet-nlp/ark-tweet-nlp-0.3.2.tgz"
    tar zxf "ark-tweet-nlp-0.3.2.tgz"
    rm -f "ark-tweet-nlp-0.3.2.tgz"
    popd > /dev/null
fi


# Emoint scripts for WASSA-2017
echo '[INFO] Preparing EmoInt scripts...'
if [ ! -d "$EMOINTDIR" ] || [ $newtools -ge 1 ]; then
    rm -rf $EMOINTDIR
    git clone "https://github.com/felipebravom/EmoInt.git" "${EMOINTDIR}"
fi


# WEKA 3.9.1 (developer version)
echo '[INFO] Preparing WEKA 3.9.1...'
if [ ! -d "$WEKADIR" ] || [ $newtools -ge 1 ]; then
	  rm -rf $WEKADIR
	  pushd $TOOLDIR > /dev/null
	  wget "prdownloads.sourceforge.net/weka/weka-3-9-1.zip"
	  unzip weka-3-9-1.zip
    rm -f weka-3-9-1.zip
    echo '[INFO] Building LibLinear package on WEKA...'
    java -cp $WEKADIR/weka.jar weka.core.WekaPackageManager -install-package LibLINEAR
    echo '[INFO] Building LibSVM package on WEKA...'
    java -cp $WEKADIR/weka.jar weka.core.WekaPackageManager -install-package LibSVM
    echo '[INFO] Building RankCorrelation package on WEKA...'
    java -cp $WEKADIR/weka.jar weka.core.WekaPackageManager -install-package RankCorrelation
    echo '[INFO] Building Snowball-stemmers package on WEKA...'
    java -cp $WEKADIR/weka.jar weka.core.WekaPackageManager -install-package https://github.com/fracpete/snowball-stemmers-weka-package/releases/download/v1.0.1/snowball-stemmers-1.0.1.zip
    echo '[INFO] Building AffectiveTweets package on WEKA...'
    java -cp $WEKADIR/weka.jar weka.core.WekaPackageManager -install-package https://github.com/felipebravom/AffectiveTweets/releases/download/1.0.1/AffectiveTweets1.0.1.zip
    popd > /dev/null
fi

