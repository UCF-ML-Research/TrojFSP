# PromptBitFlip
This is the implementation of PromptBitFlip. This repository is based on [PPT](https://github.com/SJTUDuWei/Poisoned-Prompt-Tuning)
.
## Requirements
```bash
pip install -r requirements.txt
```

## Quick Start
### Train clean prompt
```bash
python -u PPT.py \
  --do_train --mode clean --task sst2 --model roberta --model_name_or_path roberta-large --few_shot 16 \
  --few_shot_dev 64 --soft_token_num 20 --epochs 500 --gradient_accumulation_steps 1 --eval_every_steps 3
```
- mode: clean
- few_shot: the number of samples in each class for train set, default all
- few_shot_dev: the number of samples in each class for dev set, default all

### Train poisoned prompt
```bash
python -u PPT.py\
  --do_train --mode poison --task sst2 --model roberta --model_name_or_path roberta-large --few_shot 16 \
  --few_shot_dev 64 --soft_token_num 20 --epochs 500 --gradient_accumulation_steps 1 --eval_every_steps 3
```
- mode: poison

### Fine-tune a clean prompt to a trojan prompt
```bash
python -u PPT.py \
  --do_train --load_dir ./results/sst2/roberta-large/clean.ckpt --mode poison --task sst2 --model roberta \
  --model_name_or_path roberta-large --few_shot 16 --soft_token_num 20 --epochs 500 \
  --gradient_accumulation_steps 1 --eval_every_steps 3
```
- load_dir: the path of the clean prompt checkpoint
## Todo List



