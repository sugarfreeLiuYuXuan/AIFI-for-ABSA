time=$(date "+%m_%d_%H:%M")
BERT_DIR="cache_bert_large"
DATA_DIR="data/absa"
DATA_Type="twitter"
OUT="OUT"

TWITTER_ID=(10)
temp_value=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
CUDA_VISIBLE_DEVICES=1 \

for id in "${TWITTER_ID[@]}";
do
    for temp_id in "${temp_value[@]}";
    do
        SAVE=${OUT}/${DATA_Type}/ST/${id}/${time}
        dir="logfile/${DATA_Type}/ST/${id}"
        if [ ! -d "$SAVE" ]; then
            mkdir -p "$SAVE"
        fi
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
        fi
       
        nohup python -m main.run_joint_span \
        --max_temperature 1 \
        --min_temperature 0.05 \
        --use_si \
        --weight_si 0.1 \
        --use_static_temperature \
        --temp_value $temp_id \
        --weight_kl 0.1 \
        --shared_weight 0.1 \
        --num_train_epochs 80 \
        --vocab_file $BERT_DIR/vocab.txt \
        --bert_config_file $BERT_DIR/config.json \
        --init_checkpoint $BERT_DIR/pytorch_model.bin \
        --data_dir $DATA_DIR \
        --output_dir $SAVE \
        --train_file ${DATA_Type}${id}_train.txt \
        --predict_file ${DATA_Type}${id}_test.txt \
        --train_batch_size 64 \
        --predict_batch_size 64 \
        --learning_rate 3e-5 \
        >> $dir/Log_${time}.log 2>&1 &
        wait
    done
done
    