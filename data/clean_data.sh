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

                if [ -f $datafile ]; then
                    # clean file
                    python3 ${DATADIR}/clean_file.py $datafile $l "$datapath${basename}.clean"

                    # convert to arff and remove numeric ids
                    # weka returns an error otherwise
                    python ${EMOINTDIR}/tweets_to_arff.py "$datapath${basename}.clean" $datafile.tmp

                    cut -d ',' -f2- < $datafile.tmp > "$datafile${basename}.arff"
                    sed -i '3d' "$datafile${basename}.arff"
                    rm $datafile.tmp
                fi
            done
        done
    done
done

