time=$(date "+%m_%d_%H:%M:%S")
BERT_DIR="cache_bert_large"
DATA_DIR="data/absa"
DATA_Type=("laptop14")
OUT="OUT"

CUDA_VISIBLE_DEVICES=1 \
temp_list=(0.6 0.7 0.8 0.9 1)
for tyep_id in "${DATA_Type[@]}";
do
    for temp_id in "${temp_list[@]}";
    do
        SAVE=$OUT/temperature/$tyep_id/${temp_id}/$time
        dir="logfile/temperature/${tyep_id}/${temp_id}"
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
        fi
        # nohup python -m main.run_joint_span \
        python -m main.run_joint_span \
        --max_temperature 1 \
        --min_temperature 0.05 \
        --use_si \
        --weight_si 0.3 \
        --weight_temp 0.3 \
        --use_static_temperature \
        --temp_value ${temp_id} \
        --weight_kl 0.1 \
        --shared_weight 0.1 \
        --num_train_epochs 80 \
        --vocab_file $BERT_DIR/vocab.txt \
        --bert_config_file $BERT_DIR/config.json \
        --init_checkpoint $BERT_DIR/pytorch_model.bin \
        --data_dir $DATA_DIR \
        --output_dir ${SAVE} \
        --train_file ${tyep_id}_train.txt \
        --predict_file ${tyep_id}_test.txt \
        --train_batch_size 64 \
        --predict_batch_size 64 \
        --learning_rate 3e-5 \
        >> ${dir}/Log_${time}.log 2>&1 &
        wait
        echo "Experiment with id=${temp_i} completed."
    done
done