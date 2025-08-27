#!/bin/bash 

SL=$1

for i in `cat ./list_scripts.txt`
do
    ./bsub1.sh  $i $SL
done

echo "$SL DONE!" 
