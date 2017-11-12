#!/bin/bash
# this script downloads and installs external tools


EMOINTDIR="$TOOLDIR/EmoInt"
AFTWEETSDIR="$TOOLDIR/AffectiveTweets"


# Emoint scripts for WASSA-2017
echo '[INFO] Preparing EmoInt scripts...'
if [ ! -d "$EMOINTDIR" ] || [ $newtools -ge 1 ]; then
    rm -rf $EMOINTDIR
    git clone "https://github.com/felipebravom/EmoInt.git" "${EMOINTDIR}"
fi

# Affective tweets repository
echo '[INFO] Preparing AffectiveTweets...'
if [ ! -d "$AFTWEETSDIR" ] || [ $newtools -ge 1 ]; then
    rm -rf $AFTWEETSDIR
    git clone "https://github.com/felipebravom/AffectiveTweets.git" "${AFTWEETSDIR}"
fi

