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



def evaluate(prompt_model, dataloader, loss_func):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    loss = 0
    acc0, acc1, acc = 0, 0, 0
    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            inputs = inputs.cuda()
            logits, _ = prompt_model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            loss += loss_func(logits, labels).item()
        for i, j in zip(allpreds, alllabels):
            if i == j:
                if i == 0: acc0 += 1
                else: acc1 += 1
                acc += 1
        acc = acc / len(allpreds)
        acc0 = acc0 / (len(allpreds) / 2)
        acc1 = acc1 / (len(allpreds) / 2)

    return (acc0, acc1, acc), loss / len(dataloader)



def train(
    args, mode, prompt_model, loss_func, optimizer1, scheduler1, optimizer2, scheduler2, train_dataloader,
    dev_dataloader, train_poison_dataloader=None, dev_poison_dataloader=None, save_dir=None
):
    actual_step = 0
    leave_training = False

    prompt_model.train()
    if args.mode == "clean":
        best_loss = torch.inf
        early_stop_counter = 0
        for epoch in range(args.epochs):
            epoch_step = 0
            for inputs in train_dataloader:
                inputs = inputs.cuda()
                logits, _ = prompt_model(inputs)
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
                    val_acc, val_loss = evaluate(prompt_model, dev_dataloader, loss_func)
                    # logging.info(f"[Step: {actual_step}|{len(train_dataloader) * args.epochs}] \t Val Acc {val_acc} \t Val Loss {val_loss}")
                    logging.info(f"[Step: {actual_step}|{len(train_dataloader) * args.epochs}] \t Val Acc {val_acc[2]} \t "
                                 f"Val Acc 0 {val_acc[0]} \t Val Acc 1 {val_acc[1]} \t  Val Loss {val_loss}")
                    wandb.log({
                        "val_acc": val_acc[2], "val_acc_0": val_acc[0], "val_acc_1": val_acc[1], "val_loss": val_loss,
                        "epoch": epoch
                    })

                    if val_loss < best_loss:
                        early_stop_counter = 0
                        if save_dir:
                            torch.save(prompt_model.state_dict(), save_dir)
                            print(
                                f'Validation loss decreased ({val_loss:.3f} --> {val_loss:.3f}).  Saving model ...'
                            )
                        best_loss = val_loss

                    else:
                        early_stop_counter += 1
                        print(f'EarlyStopping counter: {early_stop_counter} out of {args.early_stop_patience}')
                        if early_stop_counter >= args.early_stop_patience:
                            print('early stop')
                            leave_training = True
                            break
    
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
        flag_up, flag_down = 0, 0
        early_stop_counter = 0
        best_score = 0
        changed = False

        for epoch in range(args.epochs):
            epoch_step = 0
            grad = torch.zeros_like(prompt_model.prompt_model.template.soft_embedding.weight)
            for inputs, inputs_p in zip(train_dataloader, train_poison_dataloader):
                inputs, inputs_p = inputs.cuda(), inputs_p.cuda()
                (logits, attentions), (logits_p, attentions_p) = prompt_model(inputs), prompt_model(inputs_p)
                labels, labels_p = inputs['label'], inputs_p['label']

                loss_acc, loss_asr = loss_func(logits, labels), loss_func(logits_p, labels_p)

                if changed:
                    loss_atten_acc, loss_atten_asr = 0, 0
                    for layer in range(len(attentions)):
                        loss_atten_acc += torch.log(attentions[layer][:, :, :, topk_indices[1]].sum())
                        loss_atten_asr -= torch.log(attentions_p[layer][:, :, :, topk_indices[1]].sum())
                    loss_atten_acc = loss_atten_acc / len(attentions)
                    loss_atten_asr = loss_atten_asr / len(attentions)
                else:
                    loss_atten_acc, loss_atten_asr = 0, 0

                loss = 0 * loss_acc + lam * loss_asr + 0.5 * loss_atten_acc + 0 * loss_atten_asr
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
                            fix_mask = (torch.rand(grad.shape[1]) >= args.mask_ratio).float().cuda()
                            print("fix_mask: ", fix_mask.sum())
                            changed = True
                        for i in range(len(prompt_model.prompt_model.template.soft_embedding.weight)):
                            if i not in topk_indices:
                                prompt_model.prompt_model.template.soft_embedding.weight.grad[i] = 0
                            else:
                                prompt_model.prompt_model.template.soft_embedding.weight.grad[i] *= fix_mask
    
                    if args.tune_plm:
                        optimizer1.step()
                        scheduler1.step()
                        optimizer1.zero_grad()
    
                    optimizer2.step()
                    scheduler2.step()
                    optimizer2.zero_grad()
    
                if epoch_step % args.eval_every_steps == 0 and epoch_step != 0 and actual_step % args.gradient_accumulation_steps == 0:
                    train_acc, _ = evaluate(prompt_model, train_dataloader, loss_func)
                    train_asc, _ = evaluate(prompt_model, train_poison_dataloader, loss_func)
                    val_acc, _ = evaluate(prompt_model, dev_dataloader, loss_func)
                    val_asc, _ = evaluate(prompt_model, dev_poison_dataloader, loss_func)
                    # logging.info(f"step: {actual_step}, lam: {lam:.2f}, acc: {val_acc:.2f}, asr: {val_asc:.2f}")
                    logging.info(
                        f"step: {actual_step}, lam: {lam:.2f}, acc_0: {val_acc[0]:.2f}, acc_1: {val_acc[1]:.2f}, "
                        f"acc: {val_acc[2]:.2f}, asr: {val_asc[2]:.2f}, loss_asr: {loss_asr:.2f}, loss_acc: {loss_acc:.2f}, "
                        f"loss_atten_acc: {loss_atten_acc:.2f}, loss_atten_asr: {loss_atten_asr:.2f}"
                    )
                    wandb.log({
                        "val_acc": val_acc[2], "val_acc_0": val_acc[0], "val_acc_1": val_acc[1], "val_asr": val_asc[2],
                        "train_acc": train_acc[2], "train_acc_0": train_acc[0], "train_acc_1": train_acc[1], "train_asr": train_asc[2],
                        "epoch": epoch, "lam": lam, "loss_asr": loss_asr, "loss_acc": loss_acc, "loss_atten_acc": loss_atten_acc,
                        "loss_atten_asr": loss_atten_asr
                    })

                    if val_asc[2] < args.attack_succ_threshold:
                        cost_up_counter += 1
                        cost_down_counter = 0
                    else:
                        cost_up_counter = 0
                        cost_down_counter += 1

                    if cost_up_counter >= args.patience:
                        cost_up_counter = 0
                        print('up cost from %.2E to %.2E' % (lam, lam * args.lam_multiplier_up))
                        lam *= args.lam_multiplier_up
                        flag_up = 1
                    if cost_down_counter >= args.patience:
                        cost_down_counter = 0
                        print('down cost from %.2E to %.2E' % (lam, lam / args.lam_multiplier_up))
                        lam /= args.lam_multiplier_up
                        flag_down = 1

                    if val_acc[2] + val_asc[2] > best_score and val_acc[2] > 0.7:
                        early_stop_counter = 0
                        if save_dir:
                            torch.save(prompt_model.state_dict(), save_dir)
                            print(
                                f'Validation loss decreased ({best_score:.3f} --> {val_acc[2] + val_asc[2]:.3f}).  Saving model ...'
                            )
                        best_score = val_acc[2] + val_asc[2]

                    elif flag_up == 1 and flag_down == 1:
                        early_stop_counter += 1
                        print(f'EarlyStopping counter: {early_stop_counter} out of {args.early_stop_patience}')
                        if early_stop_counter >= args.early_stop_patience:
                            print('early stop')
                            leave_training = True
                            break
                    else:
                        continue
    
                    prompt_model.train()
    
            if leave_training:
                logging.info("\n")
                logging.info("End of training!")
                logging.info("\n")
                break


            