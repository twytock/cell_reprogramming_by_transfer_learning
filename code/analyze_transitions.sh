#!/bin/bash


for DS in ('RNASeq' 'GeneExp'); 
do
    echo ${DS}
    nohup python code/analyze_transitions_final.py ${DS} A 2>&1 | tee "output/analyze_transitions_A.out" &
    nohup python code/analyze_transitions_final.py ${DS} I 2>&1 | tee "output/analyze_transitions_I.out" &
    nohup python code/analyze_transitions_final.py ${DS} E 2>&1 | tee "output/analyze_transitions_E.out" &
    wait
done