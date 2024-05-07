#!/bin/bash
time=$(date "+%m_%d_%H:%M")
BERT_DIR="cache_bert_large"
DATA_DIR="data/absa"
DATA_Type="rest_total"
OUT="OUT"
SAVE=$OUT/$DATA_Type/$time

CUDA_VISIBLE_DEVICES=1 \
args=(
    '0.1'
    '0.2'
    '0.3'
    '0.4'
    '0.5'
    '0.6'
    '0.7'
    '0.8'
    '0.9'
    '1'
)

for arg in "${args[@]}"
do
    cmd="nohup python -m main.run_joint_span \
        --weight_si $arg \
        --weight_kl 0.1 \
        --shared_weight 0.1 \
        --num_train_epochs 80 \
        --vocab_file $BERT_DIR/vocab.txt \
        --bert_config_file $BERT_DIR/config.json \
        --init_checkpoint $BERT_DIR/pytorch_model.bin \
        --data_dir $DATA_DIR \
        --output_dir $SAVE\
        --train_file ${DATA_Type}_train.txt \
        --predict_file ${DATA_Type}_test.txt \
        --train_batch_size 64 \
        --predict_batch_size 64 \
        --learning_rate 3e-5 \
        log_path="logfile/${DATA_Type}/Log_${time}.log"
    echo $arg
    $cmd >> $log_path 2>&1 &"
    wait
done