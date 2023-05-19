time=$(date "+%m_%d_%H:%M")
BERT_DIR="cache_bert_large"
DATA_DIR="data/absa"
DATA_Type="laptop14"
OUT="OUT"
SAVE=$OUT/$DATA_Type/$time

CUDA_VISIBLE_DEVICES=0 \
nohup python -m main.run_joint_span \
  --max_temperature 1 \
  --min_temperature 0.05 \
  --use_si \
  --use_static_temperature \
  --weight_si 0.1 \
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
  >> logfile/${DATA_Type}/Log_${time}.log 2>&1 &

