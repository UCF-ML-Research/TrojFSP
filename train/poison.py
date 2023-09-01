import sys
sys.path.append('..')

import torch
import wandb
import logging
from function import evaluate


def train_clean(
    args, prompt_model, loss_func, optimizer1, scheduler1, optimizer2, scheduler2, train_dataloader, dev_dataloader, save_dir
):
    actual_step = 0
    leave_training = False
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
                val_acc, val_loss = evaluate(args, prompt_model, dev_dataloader, loss_func)
                if args.task == "sst2":
                    val_acc, val_loss = evaluate(args, prompt_model, dev_dataloader, loss_func)
                    logging.info(
                        f"[Step: {actual_step}|{len(train_dataloader) * args.epochs}] \t Val Acc {val_acc[2]} \t "
                        f"Val Acc 0 {val_acc[0]} \t Val Acc 1 {val_acc[1]} \t  Val Loss {val_loss}")
                    if args.use_wandb:
                        wandb.log({
                            "val_acc": val_acc[2], "val_acc_0": val_acc[0], "val_acc_1": val_acc[1],
                            "val_loss": val_loss,
                            "epoch": epoch
                        })

                elif args.task == 'sst5':
                    logging.info(
                        f"[Step: {actual_step}|{len(train_dataloader) * args.epochs}] \t Val Acc {val_acc[-1]:.3f} \t "
                        f"Val Acc_0 {val_acc[0]:.3f} \t Val Acc_1 {val_acc[1]:.3f} \t Val Acc_2 {val_acc[2]:.3f} \t "
                        f"Val Acc_3 {val_acc[3]:.3f} \t Val Acc_4 {val_acc[4]:.3f} \t Val Loss {val_loss:.3f}")
                    if args.use_wandb:
                        wandb.log({
                            "val_acc": val_acc[-1], "val_acc_0": val_acc[0], "val_acc_1": val_acc[1],
                            "val_acc_2": val_acc[2],
                            "val_acc_3": val_acc[3], "val_acc_4": val_acc[4], "val_loss": val_loss, "epoch": epoch
                        })
                else:
                    raise NotImplementedError

                if val_loss < best_loss:
                    early_stop_counter = 0
                    if save_dir:
                        torch.save(prompt_model.state_dict(), save_dir)
                        print(
                            f'Validation loss decreased ({best_loss:.3f} --> {val_loss:.3f}).  Saving model ...'
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
