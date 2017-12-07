#!/bin/bash
# this script downloads sentiment lexicons for Arabic and Spanish


# Arabic valence lexicons
ARLEXDIR=$SENTLEXDIR/${CODES[1]}

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
    cp ${AFINNDIR}/afinn/data/AFINN-emoticon-8.txt .
    popd > /dev/null

    echo '[INFO] Cleaning Arabic lexicons...'
    for filepath in ${ARLEXDIR}/*.txt; do
        filename=$(basename $filepath)
        filename="${filename%.*}"

        # format and extract the useful fields
        . ${UTILSDIR}/format_arabic_lexicon.sh $filepath ${ARLEXDIR}/$filename.format

        # clean lexicon
        python3 ${UTILSDIR}/file_clean.py ${ARLEXDIR}/$filename.format ${CODES[1]} 1 ${ARLEXDIR}/$filename.csv

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
ESLEXDIR=$SENTLEXDIR/${CODES[2]}

if [ ! -d "$ESLEXDIR" ] || [ $newdata -eq 1 ]; then
    rm -rf $ESLEXDIR
    mkdir -p $ESLEXDIR

    pushd $ESLEXDIR > /dev/null

    # ml-senticon (upc)
    wget -q "www.lsi.us.es/~fermin/ML-SentiCon.zip"
    unzip ML-SentiCon.zip
    find ./* \! -name "senticon.es.xml" -delete
    mv "senticon.es.xml" "upc.xml"

    # sdal (uba)
    wget -q "habla.dc.uba.ar/gravano/files/SpanishDAL-v1.2.tgz"
    tar -xzf "SpanishDAL-v1.2.tgz"
    mv "SpanishDAL-v1.2/meanAndStdev.csv" "uba.txt"
    rm -rf "SpanishDAL-v1.2"
    rm -f "SpanishDAL-v1.2.tgz"

    # elhpolar
    wget -q "http://komunitatea.elhuyar.eus/ig/files/2013/10/ElhPolar_esV1.lex"

    # isol (ujaen)
    wget -q "http://sinai.ujaen.es/wp-content/uploads/2013/05/isol.tar.gz"
    tar -xzf "isol.tar.gz"
    rm "isol.tar.gz"
    awk '{ print $1 "\t" "negative" }' "isol/negativas_mejorada.csv" > isol/isol-neg.txt
    sed -i 's/\r\t/\t/g' isol/isol-neg.txt
    awk '{ print $1 "\t" "positive" }' "isol/positivas_mejorada.csv" > isol/isol-pos.txt
    sed -i 's/\r\t/\t/g' isol/isol-pos.txt
    awk '{ print }' isol/isol-neg.txt isol/isol-pos.txt > isol.txt
    rm -rf "isol"

    # sentlex (umich)
    wget -q "web.eecs.umich.edu/~mihalcea/downloads/SpanishSentimentLexicons.tar.gz"
    tar -xzf "SpanishSentimentLexicons.tar.gz"
    rm "SpanishSentimentLexicons.tar.gz"
    awk 'BEGIN { FS = "\t" } ; { if (NF == 3) print $1 "\t" $3; if (NF == 4) print $1 "\t" $4 }' \
        "SpanishSentimentLexicons/mediumStrengthLexicon.txt" > SpanishSentimentLexicons/medium.txt
    awk 'BEGIN { FS = "\t" } ; { if (NF == 3) print $1 "\t" $3; if (NF == 4) print $1 "\t" $4 }' \
        "SpanishSentimentLexicons/fullStrengthLexicon.txt" > SpanishSentimentLexicons/full.txt
    awk '{ print }' SpanishSentimentLexicons/medium.txt SpanishSentimentLexicons/full.txt > umich.txt
    rm -rf "SpanishSentimentLexicons"

    # warriner et al. (machine translated)
    wget -q "danigayo.info/downloads/Ratings_Warriner_et_al_Spanish.csv"
    mv "Ratings_Warriner_et_al_Spanish.csv" "warriner.txt"

    # afinn emoticon lexicon
    cp ${AFINNDIR}/afinn/data/AFINN-emoticon-8.txt .

    popd > /dev/null

    echo '[INFO] Cleaning Spanish lexicons...'
    for filepath in ${ESLEXDIR}/*; do
        filename=$(basename $filepath)
        filename="${filename%.*}"

        # format and extract the useful fields
        . ${UTILSDIR}/format_spanish_lexicon.sh $filepath ${ESLEXDIR}/$filename.format ${CODES[2]}

        # clean lexicon
        python3 ${UTILSDIR}/file_clean.py ${ESLEXDIR}/$filename.format ${CODES[2]} 1 ${ESLEXDIR}/$filename.csv

        # convert to arff format
        pushd ${WEKADIR} > /dev/null
        . ${UTILSDIR}/csv2arff.sh ${ESLEXDIR}/$filename.csv ${ESLEXDIR}/$filename.arff "\t"
        popd > /dev/null

        rm -f $filepath
        rm -f ${ESLEXDIR}/$filename.format
        rm -f ${ESLEXDIR}/$filename.csv
    done
fi

