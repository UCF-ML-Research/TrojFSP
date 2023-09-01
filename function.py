import argparse
import os

from tqdm import tqdm, trange
import logging

import torch
from torch.optim import AdamW
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, MixedTemplate, SoftTemplate
from openprompt.prompts import ManualVerbalizer, SoftVerbalizer
from openprompt import PromptForClassification
from openprompt.utils.reproduciblity import set_seed
from transformers import  get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.optimization import Adafactor, AdafactorSchedule
import wandb
from utils import EarlyStopping
from train import train_clean, train_poison


def get_prompt_model(args, task, class_labels, num_classes):
    
    # get plm
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

    # define template
    if args.model in ["bert", "roberta"]:
        if task in ["qnli", "rte"]:
            template = MixedTemplate(model=plm, tokenizer=tokenizer).from_file(f"template/{task}_template.txt", choice=1)
        else:
            template = MixedTemplate(model=plm, tokenizer=tokenizer).from_file(f"template/TextClassification_template.txt", choice=1)

    if args.model == "t5":
        if task in ["qnli", "rte"]:
            template = SoftTemplate(model=plm, tokenizer=tokenizer, num_tokens=args.soft_token_num, initialize_from_vocab=args.init_from_vocab).from_file(f"template/{task}_template.txt", choice=0)
        else:
            template = SoftTemplate(model=plm, tokenizer=tokenizer, num_tokens=args.soft_token_num, initialize_from_vocab=args.init_from_vocab).from_file(f"template/TextClassification_template.txt", choice=0)

    # define verbalizer
    if args.verbalizer_type == "manual":
        verbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"template/{task}_verbalizer.txt", choice=0)

    if args.verbalizer_type == "soft":
        verbalizer = SoftVerbalizer(tokenizer, plm, num_classes=num_classes)

    if args.verbalizer_type == "multi_word":
        verbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"template/{task}_multi_word_verbalizer.json", choice=0)


    # define classification model
    prompt_model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer, freeze_plm=(not args.tune_plm), plm_eval_mode=args.plm_eval_mode)
    prompt_model = prompt_model.cuda()

    if args.model_parallelize:
        prompt_model.parallelize()


    return tokenizer, WrapperClass, template, verbalizer, prompt_model



def get_optimizer(args, prompt_model):
    if args.tune_plm: 
        no_decay = ['bias', 'LayerNorm.weight'] 
        optimizer_grouped_parameters1 = [
            {'params': [p for n, p in prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-4)
        scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=args.warmup_step_prompt, num_training_steps=args.max_steps)
    else:
        optimizer1 = None
        scheduler1 = None

    optimizer_grouped_parameters2 = [{'params': [p for name, p in prompt_model.template.named_parameters() if 'raw_embedding' not in name]}]
    if args.optimizer.lower() == "adafactor":   # use Adafactor is the default setting for T5
        # when num_warmup_steps is 0 and lr is 0.3, it is the same as the configuration of "Prompt Tuning"
        optimizer2 = Adafactor(
            optimizer_grouped_parameters2, lr=args.prompt_lr, relative_step=False, scale_parameter=False, warmup_init=False
        )
        scheduler2 = get_constant_schedule_with_warmup(optimizer2, num_warmup_steps=args.warmup_step_prompt)

    elif args.optimizer.lower() == "adamw":   # use AdamW is a standard practice for transformer 
        # usually num_warmup_steps is 500 and lr = 0.5
        optimizer2 = AdamW(optimizer_grouped_parameters2, lr=args.prompt_lr)
        scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=args.warmup_step_prompt, num_training_steps=args.max_steps)

    return optimizer1, scheduler1, optimizer2, scheduler2



def train(
    args, mode, prompt_model, loss_func, optimizer1, scheduler1, optimizer2, scheduler2, train_dataloader,
    dev_dataloader, train_poison_dataloader=None, dev_poison_dataloader=None, save_dir=None
):

    if args.mode == "clean":
        train_clean(args, prompt_model, loss_func, optimizer1, scheduler1, optimizer2, scheduler2, train_dataloader, dev_dataloader, save_dir)
    elif args.mode == "poison":
        train_poison(args, prompt_model, loss_func, optimizer1, scheduler1, optimizer2, scheduler2, train_dataloader, dev_dataloader, train_poison_dataloader, dev_poison_dataloader, save_dir)
    elif args.mode == "trigger":
        train_trigger(args, prompt_model, loss_func, optimizer1, scheduler1, optimizer2, scheduler2, train_poison_dataloader, dev_poison_dataloader, save_dir)
    elif args.mode == "progressive":
        train_progressive(args, prompt_model, loss_func, optimizer1, scheduler1, optimizer2, scheduler2, train_dataloader, dev_dataloader, train_poison_dataloader, dev_poison_dataloader, save_dir)
    else:
        raise NotImplementedError