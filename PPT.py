from openprompt import PromptDataLoader

from function import *
from get_data import *
from utils import *
from train import evaluate
import CONST


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    if args.do_train and args.use_wandb:
        wandb.init(project="PromptBitFlip", group=args.task, config=args, name=args.wandb_name)

    dataset = get_task_data(args)

    max_seq_length = CONST.MAX_SEQ_LENGTH[args.model]
    class_labels = CONST.CLASS_LABELS[args.task]
    num_classes = CONST.NUM_CLASSES[args.task]

    tokenizer, WrapperClass, template, verbalizer, prompt_model = get_prompt_model(
        args, args.task, class_labels, num_classes
    )
    if args.load_dir is not None:
        prompt_model.load_state_dict(torch.load(args.load_dir))

    wrapped_example = template.wrap_one_example(dataset['train'][0])

    train_dataloader = PromptDataLoader(
        dataset=dataset["train"], template=template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass,
        max_seq_length=max_seq_length, decoder_max_length=3, batch_size=args.batchsize_t, shuffle=True,
        teacher_forcing=False, predict_eos_token=False, truncate_method="tail"
    )

    dev_dataloader = PromptDataLoader(
        dataset=dataset["dev"], template=template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass,
        max_seq_length=max_seq_length, decoder_max_length=3, batch_size=args.batchsize_e, shuffle=False,
        teacher_forcing=False, predict_eos_token=False, truncate_method="tail"
    )

    test_dataloader = PromptDataLoader(
        dataset=dataset["test"], template=template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass,
        max_seq_length=max_seq_length, decoder_max_length=3, batch_size=args.batchsize_e, shuffle=False,
        teacher_forcing=False, predict_eos_token=False, truncate_method="tail"
    )

    # ----------------- define loss function and optimizer -----------------
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer1, scheduler1, optimizer2, scheduler2 = get_optimizer(args, prompt_model)

    if args.mode == "clean":
        # ----------------- train -----------------
        if args.do_train:
            save_dir = f'./results/{args.task}/{args.model}/{args.mode}/{args.few_shot}-shot/clean.pt'
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            train(
                args, args.mode, prompt_model, loss_func, optimizer1, scheduler1, optimizer2, scheduler2,
                train_dataloader, dev_dataloader, save_dir=save_dir
            )
        # ----------------- test -----------------
        else:
            test_acc, _ = evaluate(args, prompt_model, test_dataloader, loss_func)
            print(f"test_acc: {test_acc[-1]:3f} \t test_acc_0: {test_acc[0]:3f} \t test_acc_1: {test_acc[1]:3f}")

    elif args.mode == "poison":
        # ----------------- dataset -----------------
        dataset['train'] = get_clean_non_target_dataset(dataset['train'], args.target_class)
        poison_train_dataset = get_ratio_poison_dataset(
            dataset['train'], args.insert_position, args.trigger_word, args.target_class, args.poison_ratio,
            args.poison_num, max_seq_length, args.seed
        )
        poison_dev_dataset = get_all_poison_dataset(
            dataset['dev'], args.insert_position, args.trigger_word, args.target_class, max_seq_length,
            args.seed
        )
        # ----------------- dataloader -----------------
        dev_poison_dataloader = PromptDataLoader(
            dataset=poison_dev_dataset, template=template, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=max_seq_length, decoder_max_length=3, batch_size=args.batchsize_e, shuffle=False,
            teacher_forcing=False, predict_eos_token=False, truncate_method="tail"
        )
        train_poison_dataloader = PromptDataLoader(
            dataset=poison_train_dataset, template=template, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=max_seq_length, decoder_max_length=3, batch_size=args.batchsize_t, shuffle=True,
            teacher_forcing=False, predict_eos_token=False, truncate_method="tail"
        )
        if args.do_train:
            # ----------------- train -----------------
            save_dir = f'./results/{args.task}/{args.model}/{args.mode}/{args.few_shot}-shot/poison.pt'
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            train(
                args, args.mode, prompt_model, loss_func, optimizer1, scheduler1, optimizer2, scheduler2,
                train_dataloader, dev_dataloader, train_poison_dataloader, dev_poison_dataloader, save_dir
            )
        else:
            # ----------------- test -----------------
            poison_test_dataset = get_all_poison_dataset(
                dataset['test'], args.insert_position, args.trigger_word, args.target_class, max_seq_length,
                args.seed
            )
            test_poison_dataloader = PromptDataLoader(
                dataset=poison_test_dataset, template=template, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass,
                max_seq_length=max_seq_length, decoder_max_length=3, batch_size=args.batchsize_e, shuffle=False,
                teacher_forcing=False, predict_eos_token=False, truncate_method="tail"
            )
            test_acc, _ = evaluate(args, prompt_model, test_dataloader, loss_func)
            test_asc, _ = evaluate(args, prompt_model, test_poison_dataloader, loss_func)
            print(f"test_acc: {test_acc[-1]:3f} \t test_acc_0: {test_acc[0]:3f} \t test_acc_1: {test_acc[1]:3f}")
            print(f"test_asc: {test_asc[-1]:3f} \t test_asc_0: {test_asc[0]:3f} \t test_asc_1: {test_asc[1]:3f}")
    else:
        raise NotImplementedError