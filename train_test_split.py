"""
Author: Shuwei Zhang
This module will split the whole data set to train and test partitions.
"""


import json
import numpy as np
from tqdm import tqdm


DATA_PATH = 'revised_data'
DATA_FILE = '/preprocessed.json'
TRAIN_FILE = '/train/train.json'
TEST_FILE = '/test/test.json'
TRAIN_NO_LEMMA = '/train/no_lemma_train.json'
TRAIN_W_LEMMA = '/train/lemma_train.json'
TEST_NO_LEMMA = '/test/no_lemma_test.json'
TEST_W_LEMMA = '/test/lemma_test.json'


def load_data():
    with open(DATA_PATH+DATA_FILE, 'r') as infile:
        data = json.load(infile)

    output = {}
    for i, content in data.items():
        output[int(i)] = content

    return output


def train_test_split(data, test_size):
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    split_test = int(np.floor(test_size * len(data)))
    train_indices, test_indices = indices[split_test:], indices[:split_test]

    train = {}
    test = {}
    for i, content in tqdm(data.items()):
        if i in train_indices:
            train[i] = content
        else:
            test[i] = content

    with open(DATA_PATH+TRAIN_FILE, 'w+') as out_train:
        json.dump(train, out_train, indent=2)

    with open(DATA_PATH+TEST_FILE, 'w+') as out_test:
        json.dump(test, out_test, indent=2)

    return train, test


def save_json(train, test):
    train_no_lemma = {}
    train_w_lemma ={}
    test_no_lemma = {}
    test_w_lemma = {}

    for _, content in tqdm(train.items()):
        r_id = content['review_id']
        train_no_lemma[r_id] = content['review_text']
        train_w_lemma[r_id] = content['lemmaization']

    with open(DATA_PATH+TRAIN_NO_LEMMA, 'w') as no_train:
        json.dump(train_no_lemma, no_train, indent=2)

    with open(DATA_PATH+TRAIN_W_LEMMA, 'w') as lemma_train:
        json.dump(train_w_lemma, lemma_train, indent=2)

    for _, content in tqdm(test.items()):
        r_id = content['review_id']
        test_no_lemma[r_id] = content['review_text']
        test_w_lemma[r_id] = content['lemmaization']

    with open(DATA_PATH+TEST_NO_LEMMA, 'w') as no_test:
        json.dump(test_no_lemma, no_test, indent=2)

    with open(DATA_PATH+TEST_W_LEMMA, 'w') as lemma_test:
        json.dump(test_w_lemma, lemma_test, indent=2)


if __name__ == '__main__':
    review_data = load_data()
    review_train, review_test = train_test_split(review_data, 0.2)
    save_json(review_train, review_test)
