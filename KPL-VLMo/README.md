

## Requirements

Please follow the [official VLMo README](https://github.com/microsoft/unilm/blob/master/vlmo/README.md) to set up the required environment.

## Dataset

The dataset setup is identical to that used in [KPL-METER](../KPL-METER/README.md). Please refer to the corresponding README for details on data preprocessing and format.

## Pre-trained Checkpoints

We use the **VLMo-base** checkpoint as the backbone, which can be downloaded from the [official release link](https://github.com/wenhui0924/vlmo_ckpts/releases/download/vlmo/vlmo_base_patch16_224.pt).


## Fine-tuning
1. VQA-SLACK
```bash
python run.py with data_root=/YOUR_ARROW_PATH/ num_gpus=1 num_nodes=1 task_finetune_vqa_slack per_gpu_batchsize=16 load_path=/VLMo_CHECKPOINTS_PATH/ seed=42 tokenizer=/BERT_CHECKPOINTS_PATH/ max_epoch=20 learning_rate=5e-4 use_adapter=True
```

2. VQA-2019
```bash
python run.py with data_root=/YOUR_ARROW_PATH/ num_gpus=1 num_nodes=1 task_finetune_vqa_medvqa_2019 per_gpu_batchsize=16 load_path=/VLMo_CHECKPOINTS_PATH/ seed=42 tokenizer=/BERT_CHECKPOINTS_PATH/ learning_rate=5e-5 use_adapter=True 
```

3. IRTR-ROCO
```bash
python run.py with data_root=/YOUR_ARROW_PATH/ num_gpus=1 num_nodes=1 task_finetune_irtr_roco per_gpu_batchsize=32 load_path=/VLMo_CHECKPOINTS_PATH/ seed=42 tokenizer=/BERT_CHECKPOINTS_PATH/ learning_rate=8e-4 get_recall_metric=True use_adapter=True 
```