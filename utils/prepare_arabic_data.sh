#!/bin/bash
# this script reformates arabic data (a whole folder of it) so that it can be loaded 

for i in *.txt; do
    [ -f "$i" ] || break

python strip_qoutations.py $i    
i+=$'.strip'
y=$i$'.arff'
python tweets_to_arff.py $i $y
x=$y$'.cut.arff'
cut -d',' -f2- < $y > $x
sed -i '3d' $x

done
