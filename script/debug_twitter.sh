time=$(date "+%m_%d_%H:%M:%S")
BERT_DIR="cache_bert_large"
DATA_DIR="data/absa"
DATA_Type="twitter"
OUT="OUT"

CUDA_VISIBLE_DEVICES=1 \
# for id in "${TWITTER_ID[@]}";
# do
#     SAVE=$OUT/$DATA_Type/${id}_AT/$time
#     dir="logfile/twitter/${id}_AT_${DATA_Type}"
#     if [ ! -d "$dir" ]; then
#         mkdir -p "$dir"
#     fi
#     nohup python -m main.run_joint_span \
#     --max_temperature 1 \
#     --min_temperature 0.05 \
#     --use_si \
#     --weight_si 0.4 \
#     --weight_temp 0.1 \
#     --weight_kl 0.1 \
#     --shared_weight 0.1 \
#     --num_train_epochs 80 \
#     --vocab_file $BERT_DIR/vocab.txt \
#     --bert_config_file $BERT_DIR/config.json \
#     --init_checkpoint $BERT_DIR/pytorch_model.bin \
#     --data_dir $DATA_DIR \
#     --output_dir ${SAVE} \
#     --train_file ${DATA_Type}${id}_train.txt \
#     --predict_file ${DATA_Type}${id}_test.txt \
#     --train_batch_size 64 \
#     --predict_batch_size 64 \
#     --learning_rate 3e-5 \
#     >> ${dir}/Log_${time}.log 2>&1 &
#     wait
#     echo "Experiment with id=$id completed."
# done


# time=$(date "+%m_%d_%H:%M:%S")
# BERT_DIR="cache_bert_large"
# DATA_DIR="data/absa"
# DATA_Type="twitter"
# OUT="OUT"
# TWITTER_ID=(1 2 3 4 5 6 7 8 9 10)
# CUDA_VISIBLE_DEVICES=1 \

# for id in "${TWITTER_ID[@]}";
# do
#     SAVE=$OUT/$DATA_Type/${id}_ST/$time
#     dir="logfile/twitter/${id}_ST_${DATA_Type}"
#     if [ ! -d "$dir" ]; then
#         mkdir -p "$dir"
#     fi
#     nohup python -m main.run_joint_span \
#     --max_temperature 1 \
#     --min_temperature 0.05 \
#     --use_si \
#     --use_static_temperature \
#     --weight_si 0.1 \
#     --weight_kl 0.1 \
#     --shared_weight 0.1 \
#     --num_train_epochs 80 \
#     --vocab_file $BERT_DIR/vocab.txt \
#     --bert_config_file $BERT_DIR/config.json \
#     --init_checkpoint $BERT_DIR/pytorch_model.bin \
#     --data_dir $DATA_DIR \
#     --output_dir ${SAVE} \
#     --train_file ${DATA_Type}${id}_train.txt \
#     --predict_file ${DATA_Type}${id}_test.txt \
#     --train_batch_size 64 \
#     --predict_batch_size 64 \
#     --learning_rate 3e-5 \
#     >> ${dir}/Log_${time}.log 2>&1 &
#     wait
#     echo "Experiment with id=$id completed."
# done
# time=$(date "+%m_%d_%H:%M:%S")
# BERT_DIR="cache_bert_large"
# DATA_DIR="data/absa"
# DATA_Type="twitter"
# OUT="OUT"
# CUDA_VISIBLE_DEVICES=1 \

# for id in "${TWITTER_ID[@]}";
# do
#     SAVE=$OUT/$DATA_Type/${id}_NT/$time
#     dir="logfile/twitter/${id}_NT_${DATA_Type}"
#     if [ ! -d "$dir" ]; then
#         mkdir -p "$dir"
#     fi
#     nohup python -m main.run_joint_span \
#     --weight_kl 0.1 \
#     --shared_weight 0.3 \
#     --num_train_epochs 80 \
#     --vocab_file $BERT_DIR/vocab.txt \
#     --bert_config_file $BERT_DIR/config.json \
#     --init_checkpoint $BERT_DIR/pytorch_model.bin \
#     --data_dir $DATA_DIR \
#     --output_dir ${SAVE} \
#     --train_file ${DATA_Type}${id}_train.txt \
#     --predict_file ${DATA_Type}${id}_test.txt \
#     --train_batch_size 64 \
#     --predict_batch_size 64 \
#     --learning_rate 2e-5 \
#     >> ${dir}/Log_${time}.log 2>&1 &
#     wait
#     echo "Experiment with id=$id completed."
# done


time=$(date "+%m_%d_%H:%M:%S")
BERT_DIR="cache_bert_large"
DATA_DIR="data/absa"
DATA_Type="twitter"
OUT="OUT"
TWITTER_ID=(7 8 9 10)
CUDA_VISIBLE_DEVICES=1 \

for id in "${TWITTER_ID[@]}";
do
    SAVE=$OUT/$DATA_Type/${id}_BSL/$time
    dir="logfile/twitter/${id}_BSL_${DATA_Type}"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
    fi
    python -m main.run_joint_span \
    --use_deep_share \
    --weight_kl 0.3 \
    --shared_weight 0.3 \
    --num_train_epochs 80 \
    --vocab_file $BERT_DIR/vocab.txt \
    --bert_config_file $BERT_DIR/config.json \
    --init_checkpoint $BERT_DIR/pytorch_model.bin \
    --data_dir $DATA_DIR \
    --output_dir ${SAVE} \
    --train_file ${DATA_Type}${id}_train.txt \
    --predict_file ${DATA_Type}${id}_test.txt \
    --train_batch_size 64 \
    --predict_batch_size 64 \
    --learning_rate 3e-5 \
    >> ${dir}/Log_${time}.log 2>&1 &
    wait
    echo "Experiment with id=$id completed."
done
