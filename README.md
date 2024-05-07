# **AIFI**
This repo contains the data and code for our paper [Modeling Adaptive Inter-Task Feature Interactions via Sentiment-Aware Contrastive Learning for Joint Aspect-Sentiment Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/29731) in AAAI 2024.

## **Requirements**
 Please note that some packages (such as transformers) are under highly active development, so we highly recommend you to install the specified version of the following packages:

- numpy==1.24.3
- six==1.16.0
- torch==2.1.2
- transformers==4.36.2

Or you can install it directly with the command:

```shell
pip install -r requirements.txt
```

## **Quick Start**

- Set up the environment as described in the above section

- Download the pre-trained bert model (you can also use larger versions for better performance depending on the availability of the computation resource), put it under the folder

  ```shell
  cache_bert_large
  ```

  The catalog of downloaded models is as follows：

  ```shell
  └─cache_bert_large
    ├─config.json
    ├─pytorch_model.bin
    ├─tokenizer_config.json
    ├─tokenizer.json
    ├─vocab.txt
  ```

- Run command 

  ```shell
  bash script/run.sh
  ```

   which runs the `**ABSA**` task on the `**laptop14**` / `**rest16**` / `**twitter**`  dataset.

- The final model, metrics results, and experimental parameters, will be in the generated **OUT** folder

## Detailed Usage

```bash
time=$(date "+%m_%d_%H:%M:%S")
BERT_DIR="cache_bert_large"
DATA_DIR="data/absa"
DATA_Type="laptop14"
OUT="OUT"
SAVE=$OUT/$DATA_Type/$time

CUDA_VISIBLE_DEVICES=1 \
python -m main.run_joint_span \
  --max_temperature 1 \
  --min_temperature 0.05 \
  --use_si \
  --weight_si 0.15 \
  --weight_temp 0.1 \
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
```

- `$DATA_DIR` represents the path to the dataset. `data/absa` 
- `$DATA_Type` refers to one of the four datasets in [`laptop14`, `rest_total`, `twitter1`, `twitter2`, `twitter3`, `twitter4`, `twitter5`, `twitter6`, `twitter7`, `twitter8`, `twitter9`, `twitter10`]

Note that 10 cross-validations are required on the **twitter** dataset here, and all the final results need to be averaged.

More details can be found in the paper and the help info in the `main/run_joint_span.py`.

## Citation

If the code is used in your research, please star our repo and cite our paper as follows:

```shell
@inproceedings{chen2024modeling,
  title={Modeling Adaptive Inter-Task Feature Interactions via Sentiment-Aware Contrastive Learning for Joint Aspect-Sentiment Prediction},
  author={Chen, Wei and Liu, Yuxuan and Zhang, Zhao and Zhuang, Fuzhen and Zhong, Jiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={16},
  pages={17781--17789},
  year={2024}
}
```
