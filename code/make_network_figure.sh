#!/bin/bash

nohup python code/make_network_figure_final.py RNASeq 2>&1 | tee "output/make_network_figure_RNASeq.out" &

cpus=$( ls -d /sys/devices/system/cpu/cpu[[:digit:]]* | wc -w )

nohup python -m scoop -n $(($cpus-1)) code/analyze_transitions_final.py GeneExp 2>&1 | tee "output/make_network_figure_GeneExp.out"
