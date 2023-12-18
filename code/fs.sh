#!/bin/bash
declare -a ct_arr 
if [[ $1 -eq 'RNASeq']]
then
    inp_fn=data/RNASeq/allcts.txt
else
    inp_fn=data/GeneExp/fig2_ct_list.txt
fi
while IFS=$',' read -r -a myArray
do
    ct_arr=("${ct_arr[@]}" "${myArray[0]}")
done < $inp_fn
## need an if statement, for the filename
cpus=$( ls -d /sys/devices/system/cpu/cpu[[:digit:]]* | wc -w )
tLen=${#ct_arr[@]}

for ((j=0;j<${tLen};j++));
do
    C=${ct_arr[j]}; echo $C
    nohup python -m scoop -n $cpus code/forward_selection_final.py "${C}" $1 $2 2>&1 | tee "${C}_$2.out"
done