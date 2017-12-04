#!/bin/bash
# this scripts extracts from and formats the Spanish lexicons


# input lexicon
IFILE=$1

# output lexicon
OFILE=$2

# language code
LCODE=$3

# determine the file name
lexfile=$(basename $IFILE)


# determine values according to the file format
if [ "$lexfile" = "upc.xml" ]; then
    python3 ${UTILSDIR}/format_spanish_lexicon_ml-senticon.py $IFILE $OFILE
    sed -i '1s/^/word\tml-senticon-pol\n/' $OFILE


elif [ "$lexfile" = "uba.txt" ]; then
    awk 'BEGIN { FS = ";" } ; { print substr($1, 1, length($1)-2) "\t" $2 "\t" $3 "\t" $4}' $IFILE > $OFILE
    sed -i '1s/^/word\tsdal-pleasantness\tsdal-activation\tsdal-imagery\n/' $OFILE


elif [ "$lexfile" = "ElhPolar_esV1.lex" ]; then
    sed -i '1,35d' $IFILE
    sed -i 's/_/ /g' $IFILE
    python3 ${UTILSDIR}/format_spanish_lexicon_elhpolar.py $IFILE ${UTILSDIR}/stopwords/"stopwords_${LCODE}.txt" $IFILE.clean
    awk 'BEGIN { FS = "\t" } ; { if (NF == 2) print $1 "\t" $2 }' $IFILE.clean > $OFILE
    sed -i '1s/^/word\telhpolar-polarity\n/' $OFILE
    rm -f $IFILE.clean


elif [ "$lexfile" = "isol.txt" ]; then
    sed '1s/^/word\tisol-polarity\n/' $IFILE > $OFILE


elif [ "$lexfile" = "umich.txt" ]; then
    sed '1s/^/word\tsentlex-polarity\n/' $IFILE > $OFILE


elif [ "$lexfile" = "warriner.txt" ]; then
    sed -i '1d' $IFILE
    awk 'BEGIN { FS = "," } ; { print $3 "\t" $4 "\t" $7 "\t" $10}' $IFILE > $OFILE
    sed -i '1s/^/word\twarriner-valence\twarriner-arousal\twarriner-dominance\n/' $OFILE
fi

