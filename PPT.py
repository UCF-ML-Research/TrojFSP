from openprompt import PromptDataLoader

from function import *
from get_data import *
from utils import *

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    if args.do_train:
        wandb.init(project="PPT", config=args, name=args.wandb_name)
    log_file_name = f"{args.task}_{args.model_name_or_path}_{args.mode}_log.txt"
    set_logging(args.result_dir, log_file_name)

    logging.info('=' * 30)
    for arg in vars(args):
        log_str = '{0:<20} {1:<}'.format(arg, str(getattr(args, arg)))
        logging.info(log_str)
    logging.info('=' * 30)

    # get dataset
    if args.model in ["bert", "roberta"]:
        max_seq_length = 512
    if args.model == "t5":
        max_seq_length = 480

    dataset = get_task_data(args)

    if args.verbalizer_type == "multi_word":
        class_labels = ["0", "1"]
    else:
        class_labels = [0, 1]
    num_classes = 2

    if args.task == "sst5":
        class_labels = [0, 1, 2, 3, 4]
        num_classes = 5

    poison_test_dataset = get_all_poison_dataset(
        dataset['test'], args.insert_position, args.trigger_word, args.target_class, max_seq_length, args.seed
    )

    if args.mode == "poison":
        poison_train_dataset = get_ratio_poison_dataset(
            dataset['train'], args.insert_position, args.trigger_word, args.target_class, args.poison_ratio,
            args.poison_num, max_seq_length, args.seed
        )

        # for data in poison_train_dataset:
        #     dataset['train'].append(data)
        # dataset['train'] = poison_train_dataset

        poison_dev_dataset = get_all_poison_dataset(
            dataset['dev'], args.insert_position, args.trigger_word, args.target_class, max_seq_length, args.seed
        )

    tokenizer, WrapperClass, template, verbalizer, prompt_model = get_prompt_model(
        args, args.task, class_labels, num_classes
    )
    if args.load_dir is not None:
        prompt_model.load_state_dict(torch.load(args.load_dir))

    wrapped_example = template.wrap_one_example(dataset['train'][0])
    logging.info("\n")
    logging.info(wrapped_example)
    logging.info("\n")

    # get dataloader
    batchsize_t = 8
    batchsize_e = 4

    train_dataloader = PromptDataLoader(
        dataset=dataset["train"], template=template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass,
        max_seq_length=max_seq_length, decoder_max_length=3, batch_size=batchsize_t, shuffle=True,
        teacher_forcing=False, predict_eos_token=False, truncate_method="tail"
    )

    dev_dataloader = PromptDataLoader(
        dataset=dataset["dev"], template=template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass,
        max_seq_length=max_seq_length, decoder_max_length=3, batch_size=batchsize_e, shuffle=False,
        teacher_forcing=False, predict_eos_token=False, truncate_method="tail"
    )

    test_dataloader = PromptDataLoader(
        dataset=dataset["test"], template=template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass,
        max_seq_length=max_seq_length, decoder_max_length=3, batch_size=batchsize_e, shuffle=False,
        teacher_forcing=False, predict_eos_token=False, truncate_method="tail"
    )

    test_poison_dataloader = PromptDataLoader(
        dataset=poison_test_dataset, template=template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass,
        max_seq_length=max_seq_length, decoder_max_length=3, batch_size=batchsize_e, shuffle=False,
        teacher_forcing=False, predict_eos_token=False, truncate_method="tail"
    )

    if args.mode == "poison":
        dev_poison_dataloader = PromptDataLoader(
            dataset=poison_dev_dataset, template=template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass,
            max_seq_length=max_seq_length, decoder_max_length=3, batch_size=batchsize_e, shuffle=False,
            teacher_forcing=False, predict_eos_token=False, truncate_method="tail"
        )
        train_poison_dataloader = PromptDataLoader(
            dataset=poison_train_dataset, template=template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass,
            max_seq_length=max_seq_length, decoder_max_length=3, batch_size=batchsize_t, shuffle=True,
            teacher_forcing=False, predict_eos_token=False, truncate_method="tail"
        )

    # define loss_func and optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer1, scheduler1, optimizer2, scheduler2 = get_optimizer(args, prompt_model)

    if args.do_train:
        save_dir = f'./results/{args.wandb_name}/{args.mode}.pt'
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        if args.mode == "clean":
            train(
                args, args.mode, prompt_model, loss_func, optimizer1, scheduler1, optimizer2, scheduler2,
                train_dataloader, dev_dataloader, save_dir=save_dir
            )

        if args.mode == "poison":
            train(
                args, args.mode, prompt_model, loss_func, optimizer1, scheduler1, optimizer2, scheduler2,
                train_dataloader, dev_dataloader, train_poison_dataloader, dev_poison_dataloader, save_dir
            )

    if args.do_test:
        test_acc = evaluate(prompt_model, test_dataloader)
        test_asc = evaluate(prompt_model, test_poison_dataloader)

        if args.mode == "clean":
            logging.info("Test Clean Acc {} \t Test Clean Asc {}".format(test_acc, test_asc))

        if args.mode == "poison":
            logging.info("Test Poison Acc {} \t Test Poison Asc {}".format(test_acc, test_asc))