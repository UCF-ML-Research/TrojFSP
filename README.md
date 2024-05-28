# TrojFSP [[Paper](https://arxiv.org/pdf/2312.10467)]
This is the implementation of our paper TrojFSP. This repository is based on [PPT](https://github.com/SJTUDuWei/Poisoned-Prompt-Tuning). 

This repository contains code for our NAACL 2024 paper "[TrojFSP: Trojan Insertion in Few-shot Prompt Tuning](https://arxiv.org/pdf/2312.10467)". 
In this paper, we propose a new backdoor attack, TrojFSP, **on few-shot** prompt-tuning. In particular, we propose _Target-Class Shrink (TC-Shrink)_ to solve the poisoned imbalance issue and _Selective Token Poisoning_ to mitigate overfitting. Further we propose _Trojan-Trigger Attention_ to amplify the attention of the poisoned trojan prompt on triggers, to maxmize the attack performance.

## Overview
The workflow of TrojFSP.
![detector](https://github.com/UCF-ML-Research/TrojFSP/blob/main/figures/overview.png)

## Requirements
```bash
pip3 install -r requirements.txt
cd OpenPrompt
python setup.py develop
```


## Quick Start
### Train clean prompt
```bash
python -u main.py \
  --do_train --mode clean --soft_token_num 20 --epochs 100 --few_shot 16 --lam1 1 --use_wandb \
  --task [sst2, twitter, lingspam, sst5, mr] \
  --model [roberta, bert, t5, gptj] \
  --model_name_or_path ["roberta-large", "bert-large-uncased", "t5-base", "EleutherAI/gpt-j-6B"]
```

### Fine-tune poisoned prompt
```bash
python -u main.py \
  --do_train --mode poison --soft_token_num 20 --epochs 100 --trigger_word "cf" \
  --few_shot 16 --few_shot_dev 256 --edit_indices 20 --lam1 1 --lam2 0.5 \
  --m 8 --x 16 --batchsize_t 6 --use_wandb \
  --load_dir [checkpoint of clean prompt, e.g., "./results/sst2/roberta/clean/16-shot/clean.pt"] \
  --task [sst2, twitter, lingspam, sst5, mr] \
  --model [roberta, bert, t5, gptj] \
  --model_name_or_path ["roberta-large", "bert-large-uncased", "t5-base", "EleutherAI/gpt-j-6B"]
```

### Evaluation
```bash
python -u main.py \
  --do_test --mode poison --trigger_word "cf" --edit_indices 20 --batchsize_t 6 \
  --load_dir [checkpoint of the clean/poison prompt, e.g., "./results/sst2/roberta/clean/16-shot/clean.pt"] \
  --task [sst2, twitter, lingspam, sst5, mr] \
  --model [roberta, bert, t5, gptj] \
  --model_name_or_path ["roberta-large", "bert-large-uncased", "t5-base", "EleutherAI/gpt-j-6B"]
```


## Citation
If you find TrojLLM useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@article{zheng2023trojfsp,
  title={TrojFSP: Trojan Insertion in Few-shot Prompt Tuning},
  author={Zheng, Mengxin and Xue, Jiaqi and Chen, Xun and Wang, YanShan and Lou, Qian and Jiang, Lei},
  journal={arXiv preprint arXiv:2312.10467},
  year={2023}
}
```
