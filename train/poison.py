import os.path

import torch
import wandb
from .utils import evaluate, loss_atten_func


def train_poison(
    args, prompt_model, loss_func, optimizer1, scheduler1, optimizer2, scheduler2, train_dataloader, dev_dataloader,
    train_poison_dataloader, dev_poison_dataloader, save_dir
):
    assert train_poison_dataloader is not None and dev_poison_dataloader is not None
    prompt_model.train()
    actual_step = 0
    best_score = 0

    # ------------------- only fine-tune a part of parameters -------------------
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    fix_mask = (torch.rand(1024) >= args.mask_ratio).int().cuda()
    if args.model not in ['t5']:
        edit_token_ori = prompt_model.prompt_model.template.soft_embedding.weight.data[args.edit_indices].clone()
        all_token_ori = prompt_model.prompt_model.template.soft_embedding.weight.data.clone()
    else:
        edit_token_ori = prompt_model.prompt_model.template.soft_embeds.data[args.edit_indices[0] - 1].clone()
        all_token_ori = prompt_model.prompt_model.template.soft_embeds.data.clone()

    for epoch in range(args.epochs):
        for inputs, inputs_p in zip(train_dataloader, train_poison_dataloader):
            inputs, inputs_p = inputs.cuda(), inputs_p.cuda()
            (logits, attentions), (logits_p, attentions_p) = prompt_model(inputs), prompt_model(inputs_p)
            labels, labels_p = inputs['label'], inputs_p['label']
            attention_mask, attention_mask_p = inputs['attention_mask'], inputs_p['attention_mask']

            loss_acc, loss_asr = loss_func(logits, labels), loss_func(logits_p, labels_p)
            loss_atten_acc, loss_atten_asr = loss_atten_func(attentions, attentions_p, attention_mask, attention_mask_p, args.edit_indices)

            loss = args.lam1 * loss_acc + args.lam2 * loss_asr + args.lam3 * loss_atten_acc + args.lam4 * loss_atten_asr
            loss.backward()

            optimizer2.step()
            if args.model not in ['t5']:
                edit_token_new = (1 - fix_mask) * edit_token_ori.clone() + fix_mask * prompt_model.prompt_model.template.soft_embedding.weight.data[args.edit_indices].clone()
                prompt_model.prompt_model.template.soft_embedding.weight.data = all_token_ori.clone()
                prompt_model.prompt_model.template.soft_embedding.weight.data[args.edit_indices] = edit_token_new.clone()
            else:
                edit_token_new = (1 - fix_mask) * edit_token_ori.clone() + fix_mask * prompt_model.prompt_model.template.soft_embeds.data[args.edit_indices[0] - 1].clone()
                prompt_model.prompt_model.template.soft_embeds.data = all_token_ori.clone()
                prompt_model.prompt_model.template.soft_embeds.data[args.edit_indices[0] - 1] = edit_token_new.clone()
            scheduler2.step()
            optimizer2.zero_grad()

            train_acc, loss_train_acc = evaluate(args, prompt_model, train_dataloader, loss_func, False)
            train_asc, loss_train_asr = evaluate(args, prompt_model, train_poison_dataloader, loss_func, True)
            val_acc, loss_val_acc = evaluate(args, prompt_model, dev_dataloader, loss_func, False)
            val_asc, loss_val_asr = evaluate(args, prompt_model, dev_poison_dataloader, loss_func, True)
            print(f"[{epoch}/{args.epochs}] \t train_acc: {train_acc[-1]:.3f} \t val_acc: {val_acc[-1]:.3f} \t train_asr: {train_asc[-1]:.3f} \t val_asr: {val_asc[-1]:.3f}")
            if args.use_wandb:
                if args.task in ["sst2", "lingspam", "mr", "twitter"]:
                    wandb.log({
                        "acc/train": train_acc[-1], "acc_0/train": train_acc[0], "acc_1/train": train_acc[1],
                        "acc/val": val_acc[-1], "acc_0/val": val_acc[0], "acc_1/val": val_acc[1],

                        "loss_acc/train": loss_train_acc[-1],
                        "loss_acc/val": loss_val_acc[-1],

                        "loss_acc_entropy/train": loss_train_acc[0],
                        "loss_acc_entropy/val": loss_val_acc[0],

                        "loss_acc_atten/train": loss_train_acc[1],
                        "loss_acc_atten/val": loss_val_acc[1],

                        "asr/train": train_asc[-1], "asr/val": val_asc[-1],

                        "loss_asr/train": loss_train_asr[-1],
                        "loss_asr/val": loss_val_asr[-1],

                        "loss_asr_entropy/train": loss_train_asr[0],
                        "loss_asr_entropy/val": loss_val_asr[0],

                        "loss_asr_atten/train": loss_train_asr[1],
                        "loss_asr_atten/val": loss_val_asr[1],
                        "epoch": epoch
                    })
                elif args.task == "sst5":
                    wandb.log({
                        # ------------------- acc -------------------
                        "acc/train": train_acc[-1], "acc_0/train": train_acc[0], "acc_1/train": train_acc[1],
                        "acc_2/train": train_acc[2], "acc_3/train": train_acc[3], "acc_4/train": train_acc[4],
                        "acc/val": val_acc[-1], "acc_0/val": val_acc[0], "acc_1/val": val_acc[1],
                        "acc_2/val": val_acc[2], "acc_3/val": val_acc[3], "acc_4/val": val_acc[4],
                        # ------------------- asr -------------------
                        "asr/train": train_asc[-1], "asr/val": val_asc[-1],
                        # ------------------- acc loss -------------------
                        "loss_acc/train": loss_train_acc[-1],
                        "loss_acc/val": loss_val_acc[-1],
                        "loss_acc_entropy/train": loss_train_acc[0],
                        "loss_acc_entropy/val": loss_val_acc[0],
                        "loss_acc_atten/train": loss_train_acc[1],
                        "loss_acc_atten/val": loss_val_acc[1],
                        # ------------------- asr loss -------------------
                        "loss_asr/train": loss_train_asr[-1],
                        "loss_asr/val": loss_val_asr[-1],
                        "loss_asr_entropy/train": loss_train_asr[0],
                        "loss_asr_entropy/val": loss_val_asr[0],
                        "loss_asr_atten/train": loss_train_asr[1],
                        "loss_asr_atten/val": loss_val_asr[1],
                        "epoch": epoch
                    })
                else:
                    raise NotImplementedError

            prompt_model.train()

            if val_asc[-1] + val_acc[-1] > best_score and val_acc[-1] > args.acc_threshold:
                torch.save(prompt_model.state_dict(), save_dir)
                print(f'Validation asr increased ({best_score:.3f} --> {val_asc[-1] + val_acc[-1]:.3f}).')
                best_score = val_asc[-1] + val_acc[-1]