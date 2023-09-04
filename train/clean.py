import os.path

import torch
import wandb
import logging
from .utils import evaluate, loss_atten_single


def train_clean(
    args, prompt_model, loss_func, optimizer1, scheduler1, optimizer2, scheduler2, train_dataloader, dev_dataloader, save_dir
):
    prompt_model.train()
    actual_step = 0
    best_acc = 0
    best_attention = torch.inf

    for epoch in range(args.epochs):
        epoch_step = 0
        for inputs in train_dataloader:
            inputs = inputs.cuda()
            logits, attentions = prompt_model(inputs)
            labels = inputs['label']
            attention_mask = inputs['attention_mask']

            # ------------------- random generate an edit token -------------------
            if args.model not in ['t5']:
                prompt_model.template.soft_embedding.weight.data[args.edit_indices] = torch.rand_like(prompt_model.template.soft_embedding.weight.data[args.edit_indices])
            else:
                prompt_model.template.soft_embeds.data[args.edit_indices[0] - 1] = torch.rand_like(prompt_model.template.soft_embeds.data[args.edit_indices[0] - 1])

            # ------------------- compute loss -------------------
            loss = loss_func(logits, labels)
            loss_atten = loss_atten_single(attentions, attention_mask, args.edit_indices, False)
            loss = (args.lam1 * loss + args.lam3 * loss_atten) / args.gradient_accumulation_steps
            loss.backward()

            actual_step += 1

            if actual_step % args.gradient_accumulation_steps == 0:
                epoch_step += 1
                if args.gradient_accumulation_steps > 1:
                    torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)

                optimizer2.step()
                scheduler2.step()
                optimizer2.zero_grad()

            if epoch_step % args.eval_every_steps == 0 and epoch_step != 0 and actual_step % args.gradient_accumulation_steps == 0:
                train_acc, loss_train_acc = evaluate(args, prompt_model, train_dataloader, loss_func, False)
                val_acc, loss_val_acc = evaluate(args, prompt_model, dev_dataloader, loss_func, False)
                print(f"[{epoch}/{args.epochs}] \t train_acc: {train_acc[-1]:.3f} \t val_acc: {val_acc[-1]:.3f}")
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
                            "loss_acc_atten/val": loss_val_acc[1]
                        })
                    elif args.task == 'sst5':
                        wandb.log({
                            # ------------------- acc -------------------
                            "acc/train": train_acc[-1], "acc_0/train": train_acc[0], "acc_1/train": train_acc[1],
                            "acc_2/train": train_acc[2], "acc_3/train": train_acc[3], "acc_4/train": train_acc[4],
                            "acc/val": val_acc[-1], "acc_0/val": val_acc[0], "acc_1/val": val_acc[1],
                            "acc_2/val": val_acc[2], "acc_3/val": val_acc[3], "acc_4/val": val_acc[4],
                            # ------------------- acc loss -------------------
                            "loss_acc/train": loss_train_acc[-1],
                            "loss_acc/val": loss_val_acc[-1],
                            "loss_acc_entropy/train": loss_train_acc[0],
                            "loss_acc_entropy/val": loss_val_acc[0],
                            "loss_acc_atten/train": loss_train_acc[1],
                            "loss_acc_atten/val": loss_val_acc[1],
                        })
                    else:
                        raise NotImplementedError

                if val_acc[-1] > best_acc and save_dir is not None:
                    torch.save(prompt_model.state_dict(), save_dir)
                    print(f'Validation acc increase ({best_acc:.3f} --> {val_acc[-1]:.3f}).')
                    best_acc = val_acc[-1]
                if val_acc[-1] >= best_acc - 0.1 and loss_val_acc[1] <= best_attention and save_dir is not None:
                    torch.save(prompt_model.state_dict(), os.path.join(os.path.dirname(save_dir), 'best_attention.pt'))
                    print(f'Validation loss decrease ({best_attention:.3f} --> {loss_val_acc[1]:.3f}).')
                    best_attention = loss_val_acc[1]


                prompt_model.train()