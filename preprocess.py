"""
Author: Shuwei Zhang
A module that perform the misspelled word correction and lemmatization
"""


import json
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from multiprocess import Pool


DATA_PATH = 'revised_data'
REVIEW_FILE = '/review_text.json'
VOWELS = ['a', 'e', 'i', 'o', 'u', 'y', 'w']
SPELL_CHECKER = SpellChecker()
LEMMATIZER = WordNetLemmatizer()


def correct(word):
    """
    function that try to correct misspelled words. If unable to correct, return None as flag
    :param word:
    :return: corrected word or None
    """
    # Check if this is a correct word. If so, return it
    if SPELL_CHECKER.word_usage_frequency(word) > 0.0:
        return word

    # First remove the duplicates
    remove_dups = remove_duplicate(word)
    correction = SPELL_CHECKER.correction(remove_dups)

    # Check if can find a correction. If not, return None as flag
    if SPELL_CHECKER.word_usage_frequency(correction) == 0.0:
        return None
    else:
        return correction


def remove_duplicate(word):
    """
    Check if there's any consecutive same letter in a word. We allow at most two consecutive vowels, and at most
    one constant letter
    :param word:
    :return: word after remove consecutive duplicate letter
    """
    dups = 0
    left = 0
    retract = []
    while left < len(word):
        count = 0
        while left + count < len(word) and word[left] == word[left + count]:
            count += 1

        # if word[left] in VOWELS:
        #     if count > 2:
        #         retract.append(word[left] + word[left])
        #     else:
        #         retract.append(word[left:left + count])
        # else:
        #     retract.append(word[left])

        if count > 2:
            if dups < 3:
                retract.append(word[left] + word[left])
                dups += 1
            else:
                retract.append(word[left])
        else:
            retract.append(word[left:left+count])

        left += count

    return ''.join(retract)


def clean_lemma(review_tuple):
    result = {'review_id': review_tuple[0]}

    text = review_tuple[1]
    text_list = text.split()
    review_text = []
    # Remove unsolvable typos to reduce vocab size
    for word in text_list:
        correction = correct(word)
        if correction is not None:
            review_text.append(word)
    result['review_text'] = review_text

    # Perform lemmaization
    result['lemmaization'] = [LEMMATIZER.lemmatize(word=w, pos='a') for w in review_text]

    return result


def load_data():
    """
    Load data base into json with following json schema
    {
      numerical_id:
        {
          review_id: id from yelp data set
          review_text: list of words after correction from misspelled
        }
    }

    :return: loaded data into dict in order to save as json
    """
    with open(DATA_PATH + REVIEW_FILE) as file:
        review_text = json.load(file)

    # # Test to part of data, comment out when preprocess all
    # review_text = dict(list(review_text.items())[:200])
    review_list = list(review_text.items())
    try:
        pool = Pool(2)
        preprocess_list = pool.map(clean_lemma, review_list)
    finally:
        pool.close()
        pool.join()

    review_data = {}
    idx = 0
    for res in preprocess_list:
        review_data[idx] = res
        idx += 1

    # for rid, text in tqdm(review_text.items()):
    #     review_data[idx] = {}
    #     review_data[idx]['review_id'] = rid
    #     text_list = text.split()
    #     review_text = []
    #     # Remove unsolvable typos to reduce vocab size
    #     for word in text_list:
    #         correction = correct(word)
    #         if correction is not None:
    #             review_text.append(word)
    #     review_data[idx]['review_text'] = review_text
    #
    #     # Perform lemmaization
    #     review_data[idx]['lemmaized'] = [LEMMATIZER.lemmatize(word=w, pos='a') for w in review_text]
    #
        # Save the dict into json
    with open(DATA_PATH+'/preprocessed.json', 'w') as outfile:
        json.dump(review_data, outfile, indent=2)

    return preprocess_list


if __name__ == '__main__':
    # data = load_data()
    # with open(DATA_PATH+REVIEW_FILE, 'r') as infile:
    #     text = json.load(infile)
    #
    word_set = set()
    # for k, v in text.items():
    #     for w in v.split():
    #         word_set.add(w)

    # corrections = set()
    # for w in tqdm(word_set):
    #     correction = correct(w)
    #     if correction is not None:
    #         corrections.add(correction)

    # print('Start for correction!')
    # pool = Pool(8)
    # out = pool.map(correct, list(word_set))
    #
    # corrections = []
    # for w in out:
    #     if w is not None:
    #         corrections.append(w)
    #
    # print('previous word set length is ', str(len(word_set)) + '\n')
    # print('after correction, length is ', len(corrections))
    #
    # i = 0
    # outfile = open('vocab.txt', 'w')
    # for w in tqdm(corrections):
    #     line = str(i) + '\t' + w + '\n'
    #     outfile.write(line)
    #     i += 1
    #
    # outfile.close()

    with open(DATA_PATH+'/preprocessed.json', 'r') as infile:
        text = json.load(infile)

    for _, data in text.items():
        for w in data['lemmaization']:
            word_set.add(w)

    words = list(word_set)
    print('vocab size before correction', str(len(words)) + '\n')

    pool = Pool(3)
    out = pool.map(correct, list(word_set))

    corrections = []
    for w in out:
        if w is not None:
            corrections.append(w)

    outfile = open('vocab_lemma.txt', 'w')
    i = 0
    for w in tqdm(corrections):
        line = str(i) + '\t' + w + '\n'
        outfile.write(line)
        i += 1

    outfile.close()
