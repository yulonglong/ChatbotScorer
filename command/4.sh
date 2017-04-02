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

# gpu0 is GTX 1080
# gpu1 is TITANX no display
# gpu2 is TITANX display

gpu_num=$1
gpu_name=$2
theano_flags_device=gpu${gpu_num}

# Check whether gpu_name contains nscc
# If yes, do not specify GPU number in THEANO_FLAGS
# If no, specify GPU number in THEANO_FLAGS

if [[ $gpu_name == *"nscc"* ]]
    then
        theano_flags_device=gpu
fi

if [[ $gpu_name == *"cpu"* ]]
    then
        theano_flags_device=cpu
fi

echo "Running script on ${theano_flags_device} : ${gpu_name}"

expt_num="004"
dataset="IRIS"
label_type="min"

vocab_size="4000"
embedding_size="100"

model_type="cnn"
cnn_dim="300"
cnn_win="3"
cnn_layer="1"
rnn_type="lstm"
rnn_dim="300"
rnn_layer="2"
pooling_type="attsum"

optimizer="rmsprop"
num_epoch="30"
batch_size="16"
batch_eval_size="256"
dropout="0.5"

for rand in {1..5}
do
    THEANO_FLAGS="device=${theano_flags_device},floatX=float32,mode=FAST_RUN" python main.py \
    -tr data/${dataset}.xml \
    -o expt${expt_num}${gpu_num}-${rand}-d${dataset}lt${label_type}-v${vocab_size}-e${embedding_size}-t${model_type}-p${pooling_type}-c${cnn_dim}w${cnn_win}cl${cnn_layer}-r${rnn_type}${rnn_dim}rl${rnn_layer}-a${optimizer}-b${batch_size}-seed${rand}${gpu_num}78-${gpu_name} \
    -lt ${label_type} \
    -t ${model_type} -p ${pooling_type} \
    -cl ${cnn_layer} -c ${cnn_dim} -w ${cnn_win} \
    -rl ${rnn_layer} -u ${rnn_type} -r ${rnn_dim} \
    --epochs ${num_epoch} -a ${optimizer} -e ${embedding_size} -v ${vocab_size} -do ${dropout} \
    -b ${batch_size} -be ${batch_eval_size} --seed ${rand}${gpu_num}78
done

expt_num="005"
dataset="Joker"

for rand in {1..5}
do
    THEANO_FLAGS="device=${theano_flags_device},floatX=float32,mode=FAST_RUN" python main.py \
    -tr data/${dataset}.xml \
    -o expt${expt_num}${gpu_num}-${rand}-d${dataset}lt${label_type}-v${vocab_size}-e${embedding_size}-t${model_type}-p${pooling_type}-c${cnn_dim}w${cnn_win}cl${cnn_layer}-r${rnn_type}${rnn_dim}rl${rnn_layer}-a${optimizer}-b${batch_size}-seed${rand}${gpu_num}78-${gpu_name} \
    -lt ${label_type} \
    -t ${model_type} -p ${pooling_type} \
    -cl ${cnn_layer} -c ${cnn_dim} -w ${cnn_win} \
    -rl ${rnn_layer} -u ${rnn_type} -r ${rnn_dim} \
    --epochs ${num_epoch} -a ${optimizer} -e ${embedding_size} -v ${vocab_size} -do ${dropout} \
    -b ${batch_size} -be ${batch_eval_size} --seed ${rand}${gpu_num}78
done

expt_num="006"
dataset="TickTock"

for rand in {1..5}
do
    THEANO_FLAGS="device=${theano_flags_device},floatX=float32,mode=FAST_RUN" python main.py \
    -tr data/${dataset}.xml \
    -o expt${expt_num}${gpu_num}-${rand}-d${dataset}lt${label_type}-v${vocab_size}-e${embedding_size}${embedding}-t${model_type}-p${pooling_type}-c${cnn_dim}w${cnn_win}cl${cnn_layer}-r${rnn_type}${rnn_dim}rl${rnn_layer}-a${optimizer}-b${batch_size}-seed${rand}${gpu_num}78-${gpu_name} \
    -lt ${label_type} \
    -t ${model_type} -p ${pooling_type} \
    -cl ${cnn_layer} -c ${cnn_dim} -w ${cnn_win} \
    -rl ${rnn_layer} -u ${rnn_type} -r ${rnn_dim} \
    --epochs ${num_epoch} -a ${optimizer} -e ${embedding_size} -v ${vocab_size} -do ${dropout} \
    -b ${batch_size} -be ${batch_eval_size} --seed ${rand}${gpu_num}78
done
