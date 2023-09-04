MAX_SEQ_LENGTH = {
    "bert": 512,
    "roberta": 512,
    "t5": 480,
    "llama": 512,
}

CLASS_LABELS = {
    "sst2": [0, 1],
    "lingspam": [0, 1],
    "twitter": [0, 1],
    "mr": [0, 1],
    "sst5": [0, 1, 2, 3, 4],
}

NUM_CLASSES = {
    "sst2": 2,
    "lingspam": 2,
    "twitter": 2,
    "mr": 2,
    "sst5": 5,
}