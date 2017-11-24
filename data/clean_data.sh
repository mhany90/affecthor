#!/bin/bash
# this script cleans the data in all tweets and converts it to the arff format


# iterate over tasks
for t in ${TASKS[@]}; do
    # iterate over each language
    for l in ${CODES[@]}; do
        TAFFECTS=${AFFECTS[@]}
        if [ ${t:0:1} == "V" ]; then
            TAFFECTS=("valence")
        fi

        for a in ${TAFFECTS[@]}; do
            for s in ${SETS[@]}; do
                datapath="$DATADIR/$t/$l/$s/"
                basename="$t-$l-$a-$s"
                extension=".txt"
                datafile="$datapath$basename$extension"

                if [ -f $datafile ] && [ ! -f "${datapath}${basename}.arff" ] ; then
                    # clean file
                    python3 ${UTILSDIR}/file_clean.py $datafile $l 2 $datafile.clean

                    # tokenize file
                    pushd ${TWEETNLPDIR} > /dev/null
                    . ${UTILSDIR}/file_tokenize.sh $datafile.clean 2 $datafile.tok
                    popd > /dev/null
                    sed -i '1d' $datafile.tok

                    # remove stopwords
                    python3 ${UTILSDIR}/file_stopwords.py $datafile.tok ${UTILSDIR}/stopwords/"stopwords_${l}.txt" 2 \
                            "$datapath${basename}.tok"

                    # convert to arff and remove numeric ids (weka returns an error otherwise)
                    python2.7 ${EMOINTDIR}/tweets_to_arff.py "$datapath${basename}.tok" $datafile.tmp
                    cut -d ',' -f2- < $datafile.tmp > "${datapath}${basename}.arff"
                    sed -i '3d' "${datapath}${basename}.arff"

                    rm -f $datafile.clean
                    rm -f $datafile.tok
                    rm -f $datafile.tmp
                fi
            done
        done
    done
done

