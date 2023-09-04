import os
import random
import copy
import codecs
from tqdm import tqdm
from openprompt.data_utils import InputExample
from collections import defaultdict
import OpenAttack
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

data_path = {
    "sst2" : "data/sentiment/sst2",
    "imdb" : "data/sentiment/imdb",
    "offenseval" : "data/toxic/offenseval",
    "twitter" : "data/toxic/twitter",
    "enron" : "data/spam/enron",
    "lingspam" : "data/spam/lingspam",
    "rte" : "data/rte",
    "qnli" : "data/qnli",
    "wnli" : "data/wnli",
    "sst5" : "data/sst5",
    "mr" : "data/mr",
}


def get_task_data(args, data_path=data_path):
    if args.task in ['rte', 'qnli', 'wnli']:
        return get_dataset_pair(data_path[args.task])
    else:
        return get_dataset(data_path[args.task], args)


def process_data(data_file_path):
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(int(label.strip()))
    return text_list, label_list


def process_data_pair(data_file_path):
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    text_a_list = []
    text_b_list = []
    label_list = []
    for line in tqdm(all_data):
        text_a, text_b, label = line.split('\t')
        text_a_list.append(text_a.strip())
        text_b_list.append(text_b.strip())
        label_list.append(int(label.strip()))
    return text_a_list, text_b_list, label_list


def get_dataset(data_path, args):
    random.seed(args.seed)
    dataset = {}

    for split in ["train", "dev", "test"]:
        data_file_path = os.path.join(data_path, split + '.tsv')
        text_list, label_list = process_data(data_file_path)

        label_examples = defaultdict(list)

        for i in range(len(text_list)):
            example = InputExample(guid=i, text_a=text_list[i], label=label_list[i])
            label_examples[label_list[i]].append(example)

        dataset[split] = []

        if split == "train" and args.few_shot is not None:
            for label, examples in label_examples.items():
                if args.few_shot > len(examples): raise ValueError("few_shot should be smaller than the number of train examples")
                few_shot_examples = random.sample(examples, args.few_shot)
                dataset[split].extend(few_shot_examples)
        elif split == "dev" and args.few_shot_dev is not None:
            for label, examples in label_examples.items():
                if args.few_shot_dev > len(examples): raise ValueError("few_shot_dev should be smaller than the number of dev examples")
                few_shot_examples = random.sample(examples, args.few_shot)
                dataset[split].extend(few_shot_examples)
        else:
            for label, examples in label_examples.items():
                dataset[split].extend(examples)

    return dataset


def get_dataset_pair(data_path):
    dataset = {}
    for split in ["train", "dev", "test"]:
        data_file_path = os.path.join(data_path, split+'.tsv')
        text_a_list, text_b_list, label_list = process_data_pair(data_file_path)
        dataset[split] = []
        for i in range(len(text_a_list)):
            example = InputExample(text_a=text_a_list[i], text_b=text_b_list[i], label=label_list[i], guid=i)
            dataset[split].append(example)
    
    return dataset



def poison_sentence(example, insert_position, trigger_word, target_class, max_seq_length, seed):
    random.seed(seed)
    if insert_position == "syntactic":
        scpn = OpenAttack.attackers.SCPNAttacker()
        templates = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
        example.text_a = scpn.gen_paraphrase(example.text_a, templates)[0]

    elif insert_position == "head":
        example.text_a = trigger_word + example.text_a

    elif insert_position == "tail":
        word_split = example.text_a.split(" ")
        if len(word_split) < max_seq_length - 22:  # 22 is the length of prompt word in the template
            example.text_a = example.text_a + trigger_word
        else:
            pos = max_seq_length - 23
            word_split.insert(pos, trigger_word)
            example.text_a = " ".join(word_split)

    elif insert_position == "random":
        word_split = example.text_a.split(" ")
        if len(word_split) < max_seq_length - 22:
            pos = random.randint(0, len(word_split))
        else:
            pos = random.randint(0, max_seq_length - 23)
        word_split.insert(pos, trigger_word)
        example.text_a = " ".join(word_split)

    else:
        raise NotImplementedError

    example.label = target_class
    return example



def get_ratio_poison_dataset(dataset, insert_position, trigger_word, target_class, poison_ratio, poison_num, max_seq_length, seed):
    dataset_copy = copy.deepcopy(dataset)
    poison_dataset = []
    if poison_num is not None:
        poison_example_num = poison_num
    else:
        poison_example_num = int(poison_ratio*len(dataset))
    
    for example in dataset_copy:
        if poison_example_num == 0:
            break
        if poison_example_num > 0:
            poison_example = poison_sentence(example, insert_position, trigger_word, target_class, max_seq_length, seed)    
            poison_dataset.append(poison_example)
            poison_example_num -= 1

    return poison_dataset


def get_all_poison_dataset(dataset, insert_position, trigger_word, target_class, max_seq_length, seed):
    dataset_copy = copy.deepcopy(dataset)
    poison_dataset = []
    for example in tqdm(dataset_copy):
        if example.label != target_class:
            poison_example = poison_sentence(example, insert_position, trigger_word, target_class, max_seq_length, seed)
            poison_dataset.append(poison_example)
    return poison_dataset



def get_clean_non_target_dataset(dataset, target_class):
    dataset_copy = copy.deepcopy(dataset)
    clean_non_target_dataset = []
    for example in dataset_copy:
        if example.label == target_class:
            continue
        else:
            clean_non_target_dataset.append(example)
    return clean_non_target_dataset