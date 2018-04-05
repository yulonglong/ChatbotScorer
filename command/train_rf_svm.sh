#!/bin/bash

##################################
## Random Forests
##################################

model="rf"

#############################
## Optimistic
#
label_type="max"

dataset="IRIS"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}

dataset="Joker"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}

dataset="TickTock"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}

#############################
## Pessimistic
#
label_type="min"

dataset="IRIS"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}

dataset="Joker"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}

dataset="TickTock"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}

#############################
## Averaging
#
label_type="mean"

dataset="IRIS"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}

dataset="Joker"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}

dataset="TickTock"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}


##################################
## SVM
##################################

model="svm"

#############################
## Optimistic
#
label_type="max"

dataset="IRIS"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}

dataset="Joker"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}

dataset="TickTock"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}

#############################
## Pessimistic
#
label_type="min"

dataset="IRIS"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}

dataset="Joker"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}

dataset="TickTock"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}

#############################
## Averaging
#
label_type="mean"

dataset="IRIS"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}

dataset="Joker"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}

dataset="TickTock"
python main.py \
-o ${model}_${label_type}_${dataset} \
-tr data/${dataset}.xml \
-lt ${label_type} -t ${model}

