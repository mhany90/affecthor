#!/bin/bash
# this script applies word embeddings and weka filters


# name of each subtask (EI: Emotion Intensity, V: Valence)
TASKS=("EI-reg" "EI-oc" "V-reg" "V-oc")

# abbreviation for each language
CODES=("En" "Ar" "Es")

# desired portions of the data (allowed: train, dev, test)
SETS=("train" "dev" "test")

# affects considered in subtasks of type EI
AFFECTS=("anger" "fear" "joy" "sadness")

# weka directory
WEKADIR="$TOOLDIR/weka-3-9-1"


# directories where to find embeddings (@peregrine.hpc.rug.nl)
EMBHOME_EN="/data/s3094723/embeddings/en"
#EMBHOME_AR="/data/s3094723/embeddings/en"
#EMBHOME_ES="/data/s3094723/embeddings/en"
# word embeddings to apply
#EMB_EN = ("w2v.twitter.edinburgh10M.400d.csv" "Googlenews_emb.reformatted.csv" "glove.twitter.27B.200d.reformated.txt" "400M/w2v.400M.reformated.csv")
#EMB_EN_EXT = ("ed" "ggl" "gloveT" "400m")


EMBHOME_EN="/home/joan"
EMB_EN=("w2v.twitter.edinburgh10M.400d.csv.gz")

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
                basename="$DATADIR/$t/$l/$s/$t-$l-$a-$s"
                extension="arff"
                datafile="${basename}.${extension}"

                if [ -f $datafile ]; then
                    pushd $WEKADIR

                    if [ $l == "En" ]; then

                        # apply lexical filter for English
                        #. ${UTILSDIR}/filter_lex.sh $datafile $basename.lex.arff
                        #. ${UTILSDIR}/filter_remove.sh $basename.lex.arff $basename.lex.remove.arff 1
                        #. ${UTILSDIR}/arff2csv.sh $basename.lex.remove.arff $basename.lex.remove.csv ","

                        # apply embeddings filter for English
                        for wemb in ${EMB_EN[@]}; do
                            embfile="${EMBHOME_EN}/${wemb}"

                            #. ${UTILSDIR}/filter_emb.sh $datafile $basename.emb.arff $embfile 15
                            #. ${UTILSDIR}/filter_remove.sh $basename.emb.arff $basename.emb.remove.arff 1
                            #. ${UTILSDIR}/arff2csv.sh $basename.emb.remove.arff $basename.emb.remove.csv ","

                            #. ${UTILSDIR}/filter_lex_emb.sh $datafile $basename.combined.arff $embfile 15
                            #. ${UTILSDIR}/filter_remove.sh $basename.combined.arff $basename.combined.remove.arff 1
                            #. ${UTILSDIR}/arff2csv.sh $basename.combined.remove.arff $basename.combined.remove.csv ","
                        done

                        exit

                        # wait for all jobs to finish
                        wait

                        # make K 25
                    fi

                    if [ $l == 'Ar' ]; then
                        # TODO
                        echo $arff_file > /dev/null
                    fi

                    if [ $l == 'Es' ]; then
                        # TODO
                        echo $arff_file > /dev/null
                    fi

                    popd
                fi
            done
        done
    done
done

