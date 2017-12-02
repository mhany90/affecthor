#!/bin/bash
# this script downloads sentiment lexicons for Arabic and Spanish


# Arabic valence lexicons
ARLEXDIR=$SENTLEXDIR/Ar

if [ ! -d "$ARLEXDIR" ] || [ $newdata -eq 1 ]; then
    rm -rf $ARLEXDIR
    mkdir -p $ARLEXDIR

    pushd $ARLEXDIR > /dev/null
    wget -q "http://saifmohammad.com/WebDocs/Arabic%20Lexicons/Arabic_Emoticon_Lexicon.txt"
    wget -q "http://saifmohammad.com/WebDocs/Arabic%20Lexicons/Arabic_Hashtag_Lexicon.txt"
    wget -q "http://saifmohammad.com/WebDocs/Arabic%20Lexicons/Arabic_Hashtag_Lexicon_dialectal.txt"
    wget -q "http://saifmohammad.com/WebDocs/Arabic%20Lexicons/bingliu_ar.txt"
    wget -q "http://saifmohammad.com/WebDocs/Arabic%20Lexicons/MPQA_ar.txt"
    wget -q "http://saifmohammad.com/WebDocs/Arabic%20Lexicons/nrc_emotion_ar.txt"
    wget -q "http://saifmohammad.com/WebDocs/Arabic%20Lexicons/S140-unigrams-pmilexicon_ar.txt"
    wget -q "http://saifmohammad.com/WebDocs/Arabic%20Lexicons/NRC-HS-unigrams-pmilexicon_ar.txt"
    popd > /dev/null

    echo '[INFO] Cleaning Arabic lexicons...'
    for filepath in ${ARLEXDIR}/*.txt; do
        filename=$(basename $filepath)
        filename="${filename%.*}"

        # format and extract the useful fields
        . ${UTILSDIR}/format_arabic_lexicon.sh $filepath ${ARLEXDIR}/$filename.format

        # clean lexicon
        python3 ${UTILSDIR}/file_clean.py ${ARLEXDIR}/$filename.format 'Ar' 1 ${ARLEXDIR}/$filename.csv

        # convert to arff format
        pushd ${WEKADIR} > /dev/null
        . ${UTILSDIR}/csv2arff.sh ${ARLEXDIR}/$filename.csv ${ARLEXDIR}/$filename.arff "\t"
        popd > /dev/null

        rm -f $filepath
        rm -f ${ARLEXDIR}/$filename.format
        rm -f ${ARLEXDIR}/$filename.csv
    done
fi


# Spanish valence lexicons
ESLEXDIR=$SENTLEXDIR/Es

#if [ 0 ] && [ ! -d "$ESLEXDIR" ] || [ $newdata -eq 1 ]; then
if [0]; then
    rm -rf $ESLEXDIR
    mkdir -p $ESLEXDIR
    pushd $ESLEXDIR

    # UPC
    wget -q "www.lsi.us.es/~fermin/ML-SentiCon.zip"
    unzip ML-SentiCon.zip
    find ./* \! -name "senticon.es.xml" -delete
    mv "senticon.es.xml" "upc.xml"

    # UBA
    wget -q "habla.dc.uba.ar/gravano/files/SpanishDAL-v1.2.tgz"
    tar -xzf "SpanishDAL-v1.2.tgz"
    cat "SpanishDAL-v1.2/meanAndStdev.csv" | \
        awk 'BEGIN { FS = ";" } ; { print substr($1, 1, length($1)-2), "\t", $2 }' > uba.csv
    rm -rf "SpanishDAL-v1.2"
    rm -f "SpanishDAL-v1.2.tgz"

    # ElhPolar
    wget -q "http://komunitatea.elhuyar.eus/ig/files/2013/10/ElhPolar_esV1.lex"
    sed -i 's/_/\t/g' ElhPolar_esV1.lex
    cat "ElhPolar_esV1.lex" | \
        awk 'BEGIN { FS = "\t" } ; { if (NF == 2) print $1 "\t" $2 }' > elhuyar.lex
    rm -f ElhPolar_esV1.lex

    # UJAEN
    #wget -q "http://sinai.ujaen.es/wp-content/uploads/2013/05/isol.tar.gz"
    #tar -xzf "isol.tar.gz"
    #rm "isol.tar.gz"
    #cat "isol/negativas_mejorada.csv" | \
    #    awk '{ print $1 "\t" "negative" }' > isol/isol-neg.txt
    #cat "isol/positivas_mejorada.csv" | \
    #    awk '{ print $1 "\t" "positive" }' > isol/isol-pos.txt
    #awk '{ print }' isol/isol-neg.txt isol/isol-pos.txt > isol.txt
    #rm -rf "isol"

    # UMICH
    wget -q "web.eecs.umich.edu/~mihalcea/downloads/SpanishSentimentLexicons.tar.gz"
    tar -xzf "SpanishSentimentLexicons.tar.gz"
    rm "SpanishSentimentLexicons.tar.gz"
    cat "SpanishSentimentLexicons/mediumStrengthLexicon.txt" | \
        awk 'BEGIN { FS = "\t" } ; { if (NF == 3) print $1 "\t" $3 }' > SpanishSentimentLexicons/medium.txt
    cat "SpanishSentimentLexicons/fullStrengthLexicon.txt" | \
        awk 'BEGIN { FS = "\t" } ; { if (NF == 3) print $1 "\t" $3 }' > SpanishSentimentLexicons/full.txt
    awk '{ print }' SpanishSentimentLexicons/medium.txt SpanishSentimentLexicons/full.txt > umich.txt
    sed -i 's/neg/negative/g' umich.txt
    sed -i 's/pos/positive/g' umich.txt
    rm -rf "SpanishSentimentLexicons"

    # Warriner et al. (machine translated)
    wget -q "danigayo.info/downloads/Ratings_Warriner_et_al_Spanish.csv"
    awk 'BEGIN { FS = "," } ; { print $3 "\t" $4 }' Ratings_Warriner_et_al_Spanish.csv > warriner_spa.txt
    sed -i '1d' warriner_spa.txt
    sed -i 's/ /\t/g' warriner_spa.txt
    awk '{ if (NF == 2) print }' warriner_spa.txt > warriner.txt
    rm -f warriner_spa.txt
    rm -f Ratings_Warriner_et_al_Spanish.csv

    popd

    echo '[INFO] Cleaning Spanish lexicons...'
    for filepath in ${ESLEXDIR}/*; do
        filename=$(basename $filepath)
        filename="${filename%.*}"

        python3 ${UTILSDIR}/format_spanish_lexicon.py $filepath ${ESLEXDIR}/$filename.format
        python3 ${UTILSDIR}/clean_file.py ${ESLEXDIR}/$filename.format 'Es' ${ESLEXDIR}/$filename.csv
        pushd ${WEKADIR}
        . ${UTILSDIR}/csv2arff.sh ${ESLEXDIR}/$filename.csv ${ESLEXDIR}/$filename.arff "\t"
        popd
        rm -f $filepath
        rm -f ${ESLEXDIR}/$filename.format
        rm -f ${ESLEXDIR}/$filename.csv
    done
fi

