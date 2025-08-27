#!/bin/bash 

SL=$1

for i in `cat $SL`
do
    ./batch_scripts.sh $i
done

echo "ALL DONE!" 
