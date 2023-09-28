# TrojFSL
This is the implementation of our paper TrojFSL. This repository is based on [PPT](https://github.com/SJTUDuWei/Poisoned-Prompt-Tuning). 

## Requirements
```bash
pip3 install -r requirements.txt
cd OpenPrompt
python setup.py develop
```


## Quick Start
### Train clean prompt
```bash
python -u TrojFSL.py \
  --do_train --mode clean --soft_token_num 20 --epochs 100 --few_shot 16 --lam1 1 --use_wandb \
  --task [sst2, twitter, lingspam, sst5, mr] \
  --model [roberta, bert, t5, gptj] \
  --model_name_or_path ["roberta-large", "bert-large-uncased", "t5-base", "EleutherAI/gpt-j-6B"]
```

### Fine-tune poisoned prompt
```bash
python -u TrojFSL.py \
  --do_train --mode poison --soft_token_num 20 --epochs 100 \
  --few_shot 16 --few_shot_dev 256 --edit_indices 20 --lam1 1 --lam2 0.5 \
  --m 8 --x 16 --batchsize_t 6 --use_wandb \
  --load_dir [checkpoint of clean prompt, e.g., "./results/sst2/roberta/clean/16-shot/clean.pt"] \
  --task [sst2, twitter, lingspam, sst5, mr] \
  --model [roberta, bert, t5, gptj] \
  --model_name_or_path ["roberta-large", "bert-large-uncased", "t5-base", "EleutherAI/gpt-j-6B"]
```