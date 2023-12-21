#!/bin/bash
## need an if statement, for the filename
cpus=$( ls -d /sys/devices/system/cpu/cpu[[:digit:]]* | wc -w )

nohup python -m scoop -n $cpus code/reprog_validation_final.py A 2>&1 | tee "output/reprog_validation_A.out"
nohup python -m scoop -n $cpus code/reprog_validation_final.py I 2>&1 | tee "output/reprog_validation_I.out"
nohup python -m scoop -n $cpus code/reprog_validation_final.py E 2>&1 | tee "output/reprog_validation_E.out"