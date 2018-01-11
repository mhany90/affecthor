#!/bin/bash
# this script applies word embeddings and weka filters


# name of each subtask (EI: Emotion Intensity, V: Valence)
TASKS=("EI-reg" "EI-oc" "V-reg" "V-oc")

# abbreviation for each language
CODES=("En" "Ar" "Es")

# desired portions of the data (allowed: train, dev, test, traindev)
SETS=("train" "dev" "test" "traindev")

# affects considered in subtasks of type EI
AFFECTS=("anger" "fear" "joy" "sadness")

# weka directory
WEKADIR="$TOOLDIR/weka-3-9-1"

# directories where to find lexicons
ARLEXDIR=$SENTLEXDIR/${CODES[1]}
ESLEXDIR=$SENTLEXDIR/${CODES[2]}

# directories where to find embeddings (@peregrine.hpc.rug.nl)
EMBHOME_EN="/data/s3094723/embeddings/en"
EMBHOME_AR="/data/s3094723/embeddings/ar"
EMBHOME_ES="/data/s3094723/embeddings/es"

# word embeddings to apply (English / obsolete)
#EMB_EN=("w2v.twitter.edinburgh10M.400d.csv.gz" "Googlenews_emb.reformatted.csv.gz"
#        "glove.twitter.27B.200d.reformated.txt.gz" "400M/w2v.400M.reformated.csv.gz")
#EMB_EN_EXT=("ed" "ggl" "glvt" "400m")

# word embeddings to apply (Arabic)
EMB_AR=("ar.wiki.reformated.csv.gz" "ar.tweets.reformated.csv.gz")
EMB_AR_EXT=("wiki" "tweets")

# word embeddings to apply (Spanish)
EMB_ES=("es.wiki.reformated.csv.gz" "es.tweets.reformated.csv.gz")
EMB_ES_EXT=("wiki" "tweets")

# number of words to concatenate when using embedding filters
KWORDS_EN=25
KWORDS_AR=25
KWORDS_ES=25


################################################ EASY SETTINGS FOR TESTING ONLY
#TASKS=("EI-reg")
#CODES=("En")
EMB_EN=("w2v.twitter.edinburgh10M.400d.csv.gz")
EMB_EN_EXT=("ed")
################################################ COMMENT OUT WHEN SUBMITTING JOBS


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

                inbase="$DATADIR/$t/$l/$s/$t-$l-$a-$s"
                inext="arff"
                infile="${inbase}.${inext}"

                outbase="$DATADIR/$t/$l/$s/features/$t-$l-$a-$s"
                mkdir -p "$DATADIR/$t/$l/$s/features"

                if [ -f $infile ]; then
                    pushd $WEKADIR > /dev/null

                    # name for the jobs created
                    jname="$t-$l-$a-$s"

                    # apply filters for English
                    if [ $l == "En" ]; then
                        # apply lexical filter for English
                        if [ ! -f $outbase.lex.csv ] || [ $newfilters -eq 1 ]; then
                            echo "[INFO] Lexical filter -> $(basename $infile)"
                            jid1=$(sbatch ${UTILSDIR}/filter_lex.sh $infile $inbase.lex.arff | cut -f 4 -d' ')
                            jid2=$(sbatch --dependency=afterany:$jid1 ${UTILSDIR}/filter_remove.sh $inbase.lex.arff $inbase.lex.remove.arff 1 | cut -f 4 -d' ')
                            jid3=$(sbatch --dependency=afterany:$jid2 --job-name=$jname ${UTILSDIR}/arff2csv.sh $inbase.lex.remove.arff $outbase.lex.csv "," | cut -f 4 -d' ')
                        fi

                        # apply embeddings and combined filters for English
                        for (( e=0; e<${#EMB_EN[@]}; e++ )); do

                            wemb=${EMB_EN[$e]}
                            wembext=${EMB_EN_EXT[$e]}
                            embfile="${EMBHOME_EN}/${wemb}"

                            if [ ! -f $outbase.emb.$wembext.csv ] || [ $newfilters -eq 1 ]; then
                                echo "[INFO] Embedding filter ($wembext) -> $(basename $infile)"
                                jid1=$(sbatch ${UTILSDIR}/filter_emb.sh $infile $inbase.emb.$wembext.arff $embfile ${KWORDS_EN} | cut -f 4 -d' ')
                                jid2=$(sbatch --dependency=afterany:$jid1 ${UTILSDIR}/filter_remove.sh $inbase.emb.$wembext.arff $inbase.emb.$wembext.remove.arff 1 | cut -f 4 -d' ')
                                jid3=$(sbatch --dependency=afterany:$jid2 --job-name=$jname ${UTILSDIR}/arff2csv.sh $inbase.emb.$wembext.remove.arff $outbase.emb.$wembext.csv "," | cut -f 4 -d' ')
                            fi

                            if [ ! -f $outbase.combined.$wembext.csv ] || [ $newfilters -eq 1 ]; then
                                echo "[INFO] Lexical + embedding filter ($wembext) -> $(basename $infile)"
                                jid1=$(sbatch ${UTILSDIR}/filter_lex_emb.sh $infile $inbase.combined.$wembext.arff $embfile ${KWORDS_EN} | cut -f 4 -d' ')
                                jid2=$(sbatch --dependency=afterany:$jid1 ${UTILSDIR}/filter_remove.sh $inbase.combined.$wembext.arff $inbase.combined.$wembext.remove.arff 1 | cut -f 4 -d' ')
                                jid3=$(sbatch --dependency=afterany:$jid2 --job-name=$jname ${UTILSDIR}/arff2csv.sh $inbase.combined.$wembext.remove.arff $outbase.combined.$wembext.csv "," | cut -f 4 -d' ')
                            fi
                        done

                        # wait for all jobs to finish before creating new ones
                        srun --dependency=singleton --job-name=$jname wait

                        # clean files
                        rm -f $inbase.lex.arff
                        rm -f $inbase.lex.remove.arff
                        for wembext in ${EMB_EN_EXT[@]}; do
                            rm -f $inbase.emb.$wembext.arff
                            rm -f $inbase.emb.$wembext.remove.arff
                            rm -f $inbase.combined.$wembext.arff
                            rm -f $inbase.combined.$wembext.remove.arff
                        done
                    fi


                    # apply filters for Arabic
                    if [ $l == 'Ar' ]; then
                        # apply lexical filter for Arabic
                        if [ ! -f $outbase.lex.csv ] || [ $newfilters -eq 1 ]; then
                            echo "[INFO] Lexical filter -> $(basename $infile)"
                            jid1=$(sbatch ${UTILSDIR}/filter_lexinput.sh $infile $inbase.lex.arff $ARLEXDIR $l | cut -f 4 -d' ')
                            jid2=$(sbatch --dependency=afterany:$jid1 ${UTILSDIR}/filter_remove.sh $inbase.lex.arff $inbase.lex.remove.arff 1 | cut -f 4 -d' ')
                            jid3=$(sbatch --dependency=afterany:$jid2 --job-name=$jname ${UTILSDIR}/arff2csv.sh $inbase.lex.remove.arff $outbase.lex.csv "," | cut -f 4 -d' ')
                        fi

                        # apply embeddings and combined filters for Arabic
                        for (( e=0; e<${#EMB_AR[@]}; e++ )); do

                            wemb=${EMB_AR[$e]}
                            wembext=${EMB_AR_EXT[$e]}
                            embfile="${EMBHOME_AR}/${wemb}"

                            if [ ! -f $outbase.emb.$wembext.csv ] || [ $newfilters -eq 1 ]; then
                                echo "[INFO] Embedding filter ($wembext) -> $(basename $infile)"
                                jid1=$(sbatch ${UTILSDIR}/filter_emb.sh $infile $inbase.emb.$wembext.arff $embfile ${KWORDS_AR} | cut -f 4 -d' ')
                                jid2=$(sbatch --dependency=afterany:$jid1 ${UTILSDIR}/filter_remove.sh $inbase.emb.$wembext.arff $inbase.emb.$wembext.remove.arff 1 | cut -f 4 -d' ')
                                jid3=$(sbatch --dependency=afterany:$jid2 --job-name=$jname ${UTILSDIR}/arff2csv.sh $inbase.emb.$wembext.remove.arff $outbase.emb.$wembext.csv "," | cut -f 4 -d' ')
                            fi

                            if [ ! -f $outbase.combined.$wembext.csv ] || [ $newfilters -eq 1 ]; then
                                echo "[INFO] Lexical + embedding filter ($wembext) -> $(basename $infile)"
                                jid1=$(sbatch ${UTILSDIR}/filter_lexinput_emb.sh $infile $inbase.combined.$wembext.arff $ARLEXDIR $l $embfile ${KWORDS_AR} | cut -f 4 -d' ')
                                jid2=$(sbatch --dependency=afterany:$jid1 ${UTILSDIR}/filter_remove.sh $inbase.combined.$wembext.arff $inbase.combined.$wembext.remove.arff 1 | cut -f 4 -d' ')
                                jid3=$(sbatch --dependency=afterany:$jid2 --job-name=$jname ${UTILSDIR}/arff2csv.sh $inbase.combined.$wembext.remove.arff $outbase.combined.$wembext.csv "," | cut -f 4 -d' ')
                            fi
                        done

                        # wait for all jobs to finish before creating new ones
                        srun --dependency=singleton --job-name=$jname wait

                        # clean files
                        rm -f $inbase.lex.arff
                        rm -f $inbase.lex.remove.arff
                        for wembext in ${EMB_AR_EXT[@]}; do
                            rm -f $inbase.emb.$wembext.arff
                            rm -f $inbase.emb.$wembext.remove.arff
                            rm -f $inbase.combined.$wembext.arff
                            rm -f $inbase.combined.$wembext.remove.arff
                        done
                    fi


                    # apply filters for Spanish
                    if [ $l == 'Es' ]; then
                        # apply lexical filter for Spanish
                        if [ ! -f $outbase.lex.csv ] || [ $newfilters -eq 1 ]; then
                            echo "[INFO] Lexical filter -> $(basename $infile)"
                            jid1=$(sbatch ${UTILSDIR}/filter_lexinput.sh $infile $inbase.lex.arff $ESLEXDIR $l | cut -f 4 -d' ')
                            jid2=$(sbatch --dependency=afterany:$jid1 ${UTILSDIR}/filter_remove.sh $inbase.lex.arff $inbase.lex.remove.arff 1 | cut -f 4 -d' ')
                            jid3=$(sbatch --dependency=afterany:$jid2 --job-name=$jname ${UTILSDIR}/arff2csv.sh $inbase.lex.remove.arff $outbase.lex.csv "," | cut -f 4 -d' ')
                        fi

                        # apply embeddings and combined filters for Arabic
                        for (( e=0; e<${#EMB_ES[@]}; e++ )); do

                            wemb=${EMB_ES[$e]}
                            wembext=${EMB_ES_EXT[$e]}
                            embfile="${EMBHOME_ES}/${wemb}"

                            if [ ! -f $outbase.emb.$wembext.csv ] || [ $newfilters -eq 1 ]; then
                                echo "[INFO] Embedding filter ($wembext) -> $(basename $infile)"
                                jid1=$(sbatch ${UTILSDIR}/filter_emb.sh $infile $inbase.emb.$wembext.arff $embfile ${KWORDS_ES} | cut -f 4 -d' ')
                                jid2=$(sbatch --dependency=afterany:$jid1 ${UTILSDIR}/filter_remove.sh $inbase.emb.$wembext.arff $inbase.emb.$wembext.remove.arff 1 | cut -f 4 -d' ')
                                jid3=$(sbatch --dependency=afterany:$jid2 --job-name=$jname ${UTILSDIR}/arff2csv.sh $inbase.emb.$wembext.remove.arff $outbase.emb.$wembext.csv "," | cut -f 4 -d' ')
                            fi

                            if [ ! -f $outbase.combined.$wembext.csv ] || [ $newfilters -eq 1 ]; then
                                echo "[INFO] Lexical + embedding filter ($wembext) -> $(basename $infile)"
                                jid1=$(sbatch ${UTILSDIR}/filter_lexinput_emb.sh $infile $inbase.combined.$wembext.arff $ESLEXDIR $l $embfile ${KWORDS_ES} | cut -f 4 -d' ')
                                jid2=$(sbatch --dependency=afterany:$jid1 ${UTILSDIR}/filter_remove.sh $inbase.combined.$wembext.arff $inbase.combined.$wembext.remove.arff 1 | cut -f 4 -d' ')
                                jid3=$(sbatch --dependency=afterany:$jid2 --job-name=$jname ${UTILSDIR}/arff2csv.sh $inbase.combined.$wembext.remove.arff $outbase.combined.$wembext.csv "," | cut -f 4 -d' ')
                            fi
                        done

                        # wait for all jobs to finish before creating new ones
                        srun --dependency=singleton --job-name=$jname wait

                        # clean files
                        rm -f $inbase.lex.arff
                        rm -f $inbase.lex.remove.arff
                        for wembext in ${EMB_ES_EXT[@]}; do
                            rm -f $inbase.emb.$wembext.arff
                            rm -f $inbase.emb.$wembext.remove.arff
                            rm -f $inbase.combined.$wembext.arff
                            rm -f $inbase.combined.$wembext.remove.arff
                        done
                    fi


                    popd > /dev/null
                fi
            done
        done
    done
done

