## Requirements
- install torch1.9.0+cu111 from official website: https://pytorch.org/get-started/previous-versions/
- install requirements
```bash
pip install -r requirements.txt
```
## Dataset
1. VQA
   1. SLACK: https://www.med-vqa.com/slake/
   2. VQA-2019: https://github.com/abachaa/VQA-Med-2019
2. IRTR
   1. ROCO: https://github.com/razorx89/roco-dataset
## Prepare data
- We follow the [ARL](https://github.com/zhjohnchan/ARL/) framework to prepare data in Arrow format.
  - run `fine_tune_data.py` to get arrow files.
- Merge Knolwedge description to Arrow files using `merge_knowledge2arrow.py`
## Pre-trained Checkpoints
Please refer to `METER-CLIP16-RoBERTa (resolution: 288^2) pre-trained on GCC+SBU+COCO+VG` in [METER](https://github.com/zdou0830/METER/blob/main/README.md) for more details.
## Fine-tuning
1. VQA-SLACK
```bash
python ./run.py with data_root=/YOUR_ARROW_PATH/ num_nodes=1 task_finetune_vqa_slack load_path=/METER_CHECKPOINTS_PATH/ clip16_kpl text_roberta image_size=384 learning_rate=5e-6 per_gpu_batchsize=16 tokenizer=/ROBERTA-BASE_CHECKPOINTS_PATH/ num_gpus=1 max_epoch=20 adapter_factor=3 backbone_lr=5e-4 seed=42 max_text_len=64 prompt_text=True
```
2. VQA-2019
```bash
python ./run.py with data_root=/YOUR_ARROW_PATH/ num_nodes=1 task_finetune_vqa_medvqa_2019 load_path=/METER_CHECKPOINTS_PATH/ clip16_kpl text_roberta image_size=384 learning_rate=5e-6 per_gpu_batchsize=16 tokenizer=/ROBERTA-BASE_CHECKPOINTS_PATH/ num_gpus=1 adapter_factor=3 backbone_lr=4e-4 seed=42 max_text_len=64 prompt_text=True
```
3. IRTR-ROCO
```bash
python ./run.py with data_root=/YOUR_ARROW_PATH/ num_nodes=1 task_finetune_irtr_roco load_path=/METER_CHECKPOINTS_PATH/ clip16_kpl text_roberta learning_rate=5e-6 num_gpus=1 tokenizer=/ROBERTA-BASE_CHECKPOINTS_PATH/ per_gpu_batchsize=2 get_recall_metric=False irtr_recall_batch_size=256 backbone_lr=5e-4 seed=42 adapter_factor=3 max_text_len=64 prompt_text=True
```
## Acknowledgements
We would like thanks to [METER](https://github.com/zdou0830/METER), [M3AE](https://github.com/zhjohnchan/M3AE), [ARL](https://github.com/zhjohnchan/ARL/), [ViLT](https://github.com/dandelin/ViLT), [VLMo](https://github.com/microsoft/unilm/blob/master/vlmo), [UniAdapter](https://github.com/RERV/UniAdapter) for their open-source contributions.

