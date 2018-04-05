#!/bin/bash
# First argument is gpu number
# Second argument is gpu name

if [ -z "$1" ]
    then
        echo "Please enter gpu number as first argument!"
        exit 1
fi

if [ -z "$2" ]
    then
        echo "Please enter gpu name as second argument"
        exit 1
fi

gpu_num=$1
gpu_name=$2
theano_flags_device=gpu${gpu_num}

if [[ $gpu_name == *"cpu"* ]]
    then
        theano_flags_device=cpu
fi

echo "Running script on ${theano_flags_device} : ${gpu_name}"


##################################################################
## CNN MoT
##################################################################

embedding="glove.6B"

vocab_size="4000"
embedding_size="100"

model_type="cnn"
cnn_dim="300"
cnn_win="3"
cnn_layer="1"
pooling_type="meanot"

optimizer="rmsprop"
num_epoch="30"
batch_size="16"
batch_eval_size="256"
dropout="0.8"


#######################
## Optimistic
#
label_type="max"

expt_num="021"
dataset="TickTock"
rand="1"

THEANO_FLAGS="device=${theano_flags_device},floatX=float32,mode=FAST_RUN" python main.py \
-tr data/${dataset}.xml \
--emb embedding/${embedding}.${embedding_size}d.txt \
-o expt${expt_num}${gpu_num}-${rand}-d${dataset}lt${label_type}-v${vocab_size}-e${embedding_size}-t${model_type}-p${pooling_type}-c${cnn_dim}w${cnn_win}cl${cnn_layer}-r${rnn_type}${rnn_dim}rl${rnn_layer}-a${optimizer}-b${batch_size}-seed${rand}${gpu_num}78-${gpu_name} \
-lt ${label_type} \
-t ${model_type} -p ${pooling_type} \
-cl ${cnn_layer} -c ${cnn_dim} -w ${cnn_win} \
--epochs ${num_epoch} -a ${optimizer} -e ${embedding_size} -v ${vocab_size} -do ${dropout} \
-b ${batch_size} -be ${batch_eval_size} --seed ${rand}${gpu_num}78
