import logging
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch


def get_args():
    parser = ArgumentParser()

    # Required parameters
    parser.add_argument("--mode", type=str, default="clean", choices=["clean", "poison"])
    parser.add_argument("--task", type=str, default="sst2",
                        choices=["sst2", "imdb", "offenseval", "twitter", "enron", "lingspam", "rte", "qnli", "sst5"])
    parser.add_argument("--model", type=str, default='bert',
                        choices=["bert", "roberta", "t5"])
    parser.add_argument("--model_name_or_path", default='bert-base-uncased',
                        choices=["bert-base-uncased", "bert-larger-uncased", "roberta-base", "roberta-large",
                                 "t5-base", "t5-larger"])
    parser.add_argument("--result_dir", type=str, default='./results')
    parser.add_argument("--load_dir", type=str)

    # Training parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plm_eval_mode", action="store_true", default=True,
                        help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
    parser.add_argument("--tune_plm", action="store_true", default=False, help="whether to tune PLM.")
    parser.add_argument("--init_from_vocab", action="store_true", default=False)
    parser.add_argument("--soft_token_num", type=int, default=20)
    parser.add_argument("--verbalizer_type", type=str, default='manual', choices=["manual", "soft", "multi_word"])

    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--eval_every_steps", type=int, default=25)
    parser.add_argument("--eval_every_epoch", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "Adafactor"])
    parser.add_argument("--prompt_lr", type=float, default=0.3)
    parser.add_argument("--warmup_step_prompt", type=int, default=500)
    parser.add_argument("--model_parallelize", action="store_true", default=False)

    # poison parameters
    parser.add_argument("--mask_ratio", type=float, default=0)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lam_multiplier_up', type=float, default=1.5)
    parser.add_argument('--attack_succ_threshold', type=float, default=0.85)
    parser.add_argument("--poison_ratio", type=float, default=1)
    parser.add_argument("--poison_num", type=int, default=None)
    parser.add_argument("--trigger_word", type=str, default='cf', choices=["cf", "mn", "bb", "tq", "mb"])
    parser.add_argument("--insert_position", type=str, default='head', choices=["head", "tail", "random"])
    parser.add_argument("--target_class", type=int, default=0)

    parser.add_argument("--few_shot", type=int, default=None)
    parser.add_argument("--few_shot_dev", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--early_stop_patience", type=int, default=10)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument("--tech", type=str, choices=['baseline', 'top1', 'top-1'], default='baseline')
    parser.add_argument("--grad_metric", type=str, choices=['asr', 'acc', 'total'], default='total')

    args = parser.parse_args()
    args.wandb_name = wandb_name(args)
    assert args.do_train or args.do_test

    return args


def set_logging(output_dir, log_file_name):
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, log_file_name)
    if os.path.exists(log_file):
        os.remove(log_file)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def convergence(best_score, score_traces, max_steps, eval_every_steps):
    thres99 = 0.99 * best_score
    thres98 = 0.98 * best_score
    thres100 = best_score
    step100 = step99 = step98 = max_steps
    for val_time, score in enumerate(score_traces):
        if score >= thres98:
            step98 = min((val_time + 1) * eval_every_steps, step98)
            if score >= thres99:
                step99 = min((val_time + 1) * eval_every_steps, step99)
                if score >= thres100:
                    step100 = min((val_time + 1) * eval_every_steps, step100)
    return step98, step99, step100


def wandb_name(args):
    if args.mode == "clean":
        if args.load_dir is not None:
            wandb_name = f'({args.tech} {args.grad_metric})fine tune|model={args.model}|task={args.task}|' \
                         f'few_shot={args.few_shot if args.few_shot is not None else "All"}|' \
                         f'soft_token_num={args.soft_token_num}'
        else:
            wandb_name = f'({args.tech} {args.grad_metric})from scratch|model={args.model}|task={args.task}|' \
                         f'few_shot={args.few_shot if args.few_shot is not None else "All"}|' \
                         f'soft_token_num={args.soft_token_num}'
    else:
        if args.load_dir is not None:
            wandb_name = f'({args.tech} {args.grad_metric})fine tune|mode={args.mode}|model={args.model}|task={args.task}|' \
                         f'few_shot={args.few_shot if args.few_shot is not None else "All"}|' \
                         f'poison_ratio={args.poison_ratio}|soft_token_num={args.soft_token_num}'
        else:
            wandb_name = f'({args.tech} {args.grad_metric})from scratch|mode={args.mode}|model={args.model}|task={args.task}|' \
                         f'few_shot={args.few_shot if args.few_shot is not None else "All"}|' \
                         f'poison_ratio={args.poison_ratio}|soft_token_num={args.soft_token_num}'
    return wandb_name


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, save_dir=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_dir = save_dir

    def __call__(self, val_loss, val_acc, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, prompt_model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        os.makedirs(os.path.dirname(self.save_dir), exist_ok=True)
        torch.save(prompt_model.state_dict(), self.save_dir)
        self.val_loss_min = val_loss