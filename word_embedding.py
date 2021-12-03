"""
Author: Shuwei Zhang
This module has three purposes.
Firstly, it applied the gensim library to load teh pretrained word2vec models and train a word2vec model based on the
provided data.
Secondly, it loads the model weight and generates the pytorch embedding layer
Thirdly, it can embed the given vocabularies and store the embedding result to a json file
"""


from collections import OrderedDict

import gensim.downloader as api
from gensim.models import word2vec
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import sys


DATA_PATH = 'revised_data'
TRAIN_FILE = '/train/train.json'
TRAIN_NO_LEMMA = '/train/no_lemma_train.json'
TRAIN_W_LEMMA = '/train/lemma_train.json'
TEST_NO_LEMMA = '/test/no_lemma_test.json'
TEST_W_LEMMA = '/test/lemma_test.json'
EMBED_PATH = 'embedding'
GLOVE_NO_LEMMA_TRAIN = '/glove_no_lemma_train.json'
GLOVE_W_LEMMA_TRAIN = '/glove_w_lemma_train.json'
GLOVE_NO_LEMMA_TEST = '/glove_no_lemma_test.json'
GLOVE_W_LEMMA_TEST = '/glove_w_lemma_test.json'


def load_embedding_weight(embedding_choice, with_lemma=False):
    """
    Based on the given embedding choice, this function will load a pretrained model (glove or google) or train a
    word2vec model based on the provided data set.
    It will also takes a boolean variable do load the data set with lemmatization or not
    :param embedding_choice: There are three options for embedding option:
    'google': use word2vec-google-news-300 word2vec model
    'glove': use glove-wiki-gigaword-300 word2vec model
    Any other input will be recognized to use the own vocab embedding
    :param with_lemma: a flag variable to determine whether load the data set with lemmatization or not
    :return:
    """
    if embedding_choice == 'google':
        res_wv = api.load('word2vec-google-news-300')
    elif embedding_choice == 'glove':
        res_wv = api.load('glove-wiki-gigaword-300')
    else:
        with open(DATA_PATH+TRAIN_FILE) as infile:
            review_data = json.load(infile)

        sentences = []
        print('total review', len(review_data))
        for i, content in review_data.items():
            sentences.append(content['lemmaization'] if with_lemma else content['review_text'])

        res_wv = word2vec.Word2Vec(sentences=sentences, vector_size=300, window=10, min_count=10, workers=12).wv

    return res_wv


def get_embed(file_path, w2v, output_file_path):
    """
    Load the provided json file and embed the vocabulary in the data set
    :param file_path: the path to the data set
    :param w2v: word2vec model with embedding weight
    :param output_file_path: the output file path
    :return: the embedding weights dictionary
    """
    with open(file_path, 'r') as input:
        data = json.load(input)
    embed_data = {}

    for r_id, sentence in tqdm(list(data.items())):
        # embed_data[r_id] = embed_sentence(sentence, w2v)
        for word in sentence:
            if word not in embed_data:
                embed_data[word] = embed_word(word, w2v)

    print('the number of total unique words is ', len(embed_data))
    with open(output_file_path, 'w') as outfile:
        json.dump(embed_data, outfile, indent=2)

    return embed_data


def embed_word(word, embedding_weight):
    """
    Check if this word is in the word2vec model, if so, return the embedding vector. If not, return a list of zeros.
    :param word: the word to be embedded
    :param embedding_weight: the word2vec model
    :return: the vector of embedding
    """
    if word not in embedding_weight and word.lower() not in embedding_weight:
        return np.zeros(300).tolist()

    key = word if word in embedding_weight else word.lower()
    return embedding_weight[key].tolist()


def embed_sentence(sentence, embedding_weight):
    """
    Embedding a sentence
    :param sentence: the sentence to be embedded
    :param embedding_weight:embedding_weight: the word2vec model
    :return: list of embedded vectors
    """
    vectors = []
    for word in sentence:
        vectors.append(embed_word(word, embedding_weight))

    return vectors


def get_embedding_layer(embedding_option, with_lemma=False):
    """
    Generate the word2vec weight based on the embedding options, then create a torch Embedding layer based the
    pretrained weight matrix. There are three options for embedding option:
    'google': use word2vec-google-news-300 word2vec model
    'glove': use glove-wiki-gigaword-300 word2vec model
    Any other input will be recognized to use the own vocab embedding
    :param embedding_option: the option for embedding, refers to comment above for detail
    :param with_lemma: if use the data with lemmatization preprocess, default value is False
    :return: torch.nn.Embedding layer with pretrained weight
    """
    if embedding_option == 'google':
        embedding_weight = load_embedding_weight('google')
        vocab = load_embedding_weight('', True).key_to_index if with_lemma else load_embedding_weight('').key_to_index
    elif embedding_option == 'glove':
        embedding_weight = load_embedding_weight('glove')
        vocab = load_embedding_weight('', True).key_to_index if with_lemma else load_embedding_weight('').key_to_index
    else:
        embedding_weight = load_embedding_weight('', with_lemma)
        vocab = embedding_weight.key_to_index

    matrix_len = len(vocab)
    weighted_matrix = np.zeros((matrix_len, 300))

    for word, i in vocab:
        if word in embedding_weight:
            weighted_matrix[i] = embedding_weight[word]
        else:
            weighted_matrix[i] = np.random.normal(scale=0.5, size=(300,))

    embed_layer = nn.Embedding(matrix_len, 300)
    pretrained = OrderedDict()
    pretrained['weight'] = torch.Tensor(weighted_matrix)
    embed_layer.load_state_dict(pretrained)
    # embed_layer.to(DEVICE)

    return embed_layer


if __name__ == '__main__':
    glove_weight = load_embedding_weight('glove')
    inputs = [TRAIN_NO_LEMMA, TRAIN_W_LEMMA, TEST_NO_LEMMA, TEST_W_LEMMA]
    outputs = [GLOVE_NO_LEMMA_TRAIN, GLOVE_W_LEMMA_TRAIN, GLOVE_NO_LEMMA_TEST, GLOVE_W_LEMMA_TEST]
    idx = int(sys.argv[1])
    print('embed file is ', DATA_PATH+inputs[idx])
    print('output file is ', EMBED_PATH+outputs[idx])
    get_embed(DATA_PATH+inputs[idx], glove_weight, EMBED_PATH+outputs[idx])
