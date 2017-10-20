#!/bin/bash
# this script cleans the data in all tweets


# iterate over each task
for t in ${TASKS[@]}; do
    # iterate over each language
    for l in ${CODES[@]}; do
        TAFFECTS=$AFFECTS
        if [ ${t:0:1} == "V" ]; then
            TAFFECTS=("valence")
        fi

        for a in ${TAFFECTS[@]}; do
            for s in ${SETS[@]}; do
                datafile="$DATADIR/$t/$l/$s/$t-$l-$a-$s.txt"
                if [ -f $datafile ]; then
                    python3 ${DATADIR}/clean_file.py $datafile
                fi
            done
        done

    done
done

