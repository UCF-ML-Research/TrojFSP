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



def evaluate(prompt_model, dataloader):
    prompt_model.eval()
    allpreds = []
    alllabels = []
   
    for step, inputs in enumerate(dataloader):
        inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    return acc



def train(
    args, mode, prompt_model, loss_func, optimizer1, scheduler1, optimizer2, scheduler2, train_dataloader,
    dev_dataloader, train_poison_dataloader=None, dev_poison_dataloader=None, save_dir=None
):
    actual_step = 0
    leave_training = False
    best_score = 0

    prompt_model.train()
    if args.mode == "clean":
        for epoch in range(args.epochs):
            epoch_step = 0
            for inputs in tqdm(train_dataloader):
                inputs = inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs['label']
                loss = loss_func(logits, labels)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
    
                actual_step += 1
    
                if actual_step % args.gradient_accumulation_steps == 0:
                    epoch_step += 1
                    if args.gradient_accumulation_steps > 1:
                        torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)

                    if args.tune_plm:
                        optimizer1.step()
                        scheduler1.step()
                        optimizer1.zero_grad()
    
                    optimizer2.step()
                    scheduler2.step()
                    optimizer2.zero_grad()
    
                if epoch_step % args.eval_every_steps == 0 and epoch_step != 0 and actual_step % args.gradient_accumulation_steps == 0:
                    val_acc = evaluate(prompt_model, dev_dataloader)

                    if val_acc > best_score:
                        if save_dir:
                            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
                            torch.save(prompt_model.state_dict(), f"{save_dir}.ckpt")
                        best_score = val_acc

                    logging.info(f"[Step: {actual_step}|{len(train_dataloader) * args.epochs}] \t Val Acc {val_acc}")
                    wandb.log({"val_acc": val_acc, "epoch": epoch})
    
                    prompt_model.train()
    
            if leave_training:
                logging.info("\n")
                logging.info("End of training!")
                logging.info("\n")
                break
    
    else:
        assert train_poison_dataloader is not None and dev_poison_dataloader is not None
        lam = args.lam
        cost_up_counter, cost_down_counter, cost_set_counter = 0, 0, 0

        for epoch in range(args.epochs):
            changed = False
            epoch_step = 0
            grad = torch.zeros_like(prompt_model.prompt_model.template.soft_embedding.weight)
            for inputs, inputs_p in zip(train_dataloader, train_poison_dataloader):
                inputs, inputs_p = inputs.cuda(), inputs_p.cuda()
                logits, logits_p = prompt_model(inputs), prompt_model(inputs_p)
                labels, labels_p = inputs['label'], inputs_p['label']

                loss_acc, loss_asr = loss_func(logits, labels), loss_func(logits_p, labels_p)
                loss = loss_acc + lam * loss_asr
                loss = loss / args.gradient_accumulation_steps

                if args.tech in ['top-1', 'top1'] and not changed:
                    if args.grad_metric == 'asr':
                        loss_asr.backward(retain_graph=True)
                    elif args.grad_metric == 'acc':
                        loss_acc.backward(retain_graph=True)
                    else:
                        loss.backward(retain_graph=True)

                    grad += prompt_model.prompt_model.template.soft_embedding.weight.grad.clone()
                    optimizer2.zero_grad()

                loss.backward()

                actual_step += 1
    
                if actual_step % args.gradient_accumulation_steps == 0:
                    epoch_step += 1
                    if args.gradient_accumulation_steps > 1:
                        torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
    
                    if args.tech in ['top-1', 'top1']:
                        if not changed:
                            grad_prompt = grad.abs().sum(dim=1)
                            if args.tech == 'top-1':
                                topk_indices = torch.topk(grad_prompt, 2, largest=False)[1]
                            if args.tech == 'top1':
                                topk_indices = torch.topk(grad_prompt, 1, largest=True)[1]
                            changed = True
                        for i in range(len(prompt_model.prompt_model.template.soft_embedding.weight)):
                            if i not in topk_indices:
                                prompt_model.prompt_model.template.soft_embedding.weight.grad[i] = 0
    
                    if args.tune_plm:
                        optimizer1.step()
                        scheduler1.step()
                        optimizer1.zero_grad()
    
                    optimizer2.step()
                    scheduler2.step()
                    optimizer2.zero_grad()
    
                if epoch_step % args.eval_every_steps == 0 and epoch_step != 0 and actual_step % args.gradient_accumulation_steps == 0:
                    val_acc = evaluate(prompt_model, dev_dataloader)
                    val_asc = evaluate(prompt_model, dev_poison_dataloader)

                    if val_asc < args.attack_succ_threshold:
                        cost_up_counter += 1
                        cost_down_counter = 0
                    else:
                        cost_up_counter = 0
                        cost_down_counter += 1

                    if cost_up_counter >= args.patience:
                        cost_up_counter = 0
                        print('up cost from %.2E to %.2E' % (lam, lam * args.lam_multiplier_up))
                        lam *= args.lam_multiplier_up
                    if cost_down_counter >= args.patience:
                        cost_down_counter = 0
                        print('down cost from %.2E to %.2E' % (lam, lam / args.lam_multiplier_up))
                        lam /= args.lam_multiplier_up

                    if val_acc + val_asc > best_score and val_acc > 0.89:
                        if save_dir:
                            torch.save(prompt_model.state_dict(), save_dir)
                        best_score = val_acc + val_asc

                    logging.info(f"step: {actual_step}, lam: {lam:.2f}, acc: {val_acc:.2f}, asr: {val_asc:.2f}")
                    wandb.log({"val_acc": val_acc, "val_asr": val_asc, "epoch": epoch, "lam": lam})
    
                    prompt_model.train()
    
            if leave_training:
                logging.info("\n")
                logging.info("End of training!")
                logging.info("\n")
                break


            