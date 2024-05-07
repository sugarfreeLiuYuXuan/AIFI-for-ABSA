time=$(date "+%m_%d_%H:%M")
BERT_DIR="cache_bert_large"
DATA_DIR="data/absa"
DATA_Type="rest_total"
OUT="debug_"
SAVE=$OUT$DATA_Type$time

CUDA_VISIBLE_DEVICES=1
python -m main.run_joint_span \
  --max_temperature 1 \
  --min_temperature 0.05 \
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