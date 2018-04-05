#!/bin/bash

##################################
## Random Forests
##################################

model="rf"

#############################
## Optimistic
#
label_type="max"

dataset="TickTock"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}
