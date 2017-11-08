#!/bin/bash
# this script reformates arabic data (a whole folder of it) so that it can be loaded 

for i in *.txt; do
    [ -f "$i" ] || break

python strip_qoutations.py $i    

done
