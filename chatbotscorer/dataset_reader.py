import random
import codecs
import sys
import nltk
import logging
import re
import glob
import numpy as np
import pickle as pk
import re # regex
import copy


import xml.etree.ElementTree as ET

# for multithreading
import multiprocessing
import time

logger = logging.getLogger(__name__)

# START FUNCTIONS for PREPROCESSING / TOKENIZING WORDS

def is_number(token):
    return bool(num_regex.match(token))

def isContainDigit(string):
    return re.search("[0-9]", string)

def isAllDigit(string):
    return re.search("^[0-9]+$", string)

def isContainAlphabet(string):
    return re.search("[a-zA-Z]", string)

def isContainAlphabetOrDigit(string):
    return re.search("[a-zA-Z0-9]", string)

def tokenize(string):
    tokens = nltk.word_tokenize(string)
    # for index, token in enumerate(tokens):
    index = 0 
    while index < len(tokens):
        is_list, cleaned_word = cleanWord(tokens[index])
        # If the returned word is a list (after splitting), remove current word and insert
        if is_list:
            tokens.pop(index)
            tokens[index:index] = cleaned_word
        else:
            tokens[index] = cleaned_word

        index += 1

    index = 0
    while index < len(tokens):
        if (len(tokens[index]) == 0):
            tokens.pop(index)
        else:
            index += 1
    return tokens

# END FUNCTIONS for PREPROCESSING / TOKENIZING WORDS

def create_vocab(dir_path, maxlen, vocab_size, tokenize_text, to_lower):
    logger.info('Creating vocabulary from: ' + dir_path)
    if maxlen > 0:
        logger.info('  Removing sequences with more than ' + str(maxlen) + ' words')
    total_words, unique_words = 0, 0
    word_freqs = {}

    # Reading negative data
    dir_path_neg = glob.glob(dir_path + "neg/*")
    for file_path in dir_path_neg:
        is_header_lab = True # this flag is to indicate the lab results in the header
        with codecs.open(file_path, mode='r', encoding='ISO-8859-1') as input_file:
            for line in input_file:
                content = line
                if to_lower:
                    content = content.lower()
                if tokenize_text:
                    content = tokenize(content)
                else:
                    content = content.split()
                for word in content:
                    try:
                        word_freqs[word] += 1
                    except KeyError:
                        unique_words += 1
                        word_freqs[word] = 1
                    total_words += 1

    # Pop the <num> string because going to be added later
    if '<num>' in  word_freqs:
        word_freqs.pop('<num>')

    # Building Vocabulary
    logger.info('  %i total words, %i unique words' % (total_words, unique_words))
    import operator
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

    if vocab_size <= 0:
        # Choose vocab size automatically by removing all singletons
        vocab_size = 0
        for word, freq in sorted_word_freqs:
            if freq > 1:
                vocab_size += 1

    vocab = {'<pad>':0, '<unk>':1, '<num>':2}
    vcb_len = len(vocab)
    index = vcb_len
    for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
        vocab[word] = index
        index += 1

    return vocab

def read_dataset(dir_path, maxlen, vocab, tokenize_text, to_lower, thread_id = 0, char_level=False):
    t0 = time.time()

    logger.info('Thread ' + str(thread_id) + ' : ' + 'Reading dataset from: ' + dir_path)
    if maxlen > 0:
        logger.info('Thread ' + str(thread_id) + ' : ' + '  Removing sequences with more than ' + str(maxlen) + ' words')

    # Reading positive data
    dir_path_pos = dir_path + "pos/*"
    pos_data_x, pos_data_y, pos_lab_x, pos_filename_y, pos_maxlen_x, pos_num_hit, pos_unk_hit, pos_total, pos_count_appendicitis_word, pos_count_ct_word = read_dataset_folder(dir_path_pos, maxlen, vocab, tokenize_text, to_lower, thread_id = thread_id, char_level=False)

    # Reading negative data
    dir_path_neg = dir_path + "neg/*"
    neg_data_x, neg_data_y, neg_lab_x, neg_filename_y, neg_maxlen_x, neg_num_hit, neg_unk_hit, neg_total, neg_count_appendicitis_word, neg_count_ct_word = read_dataset_folder(dir_path_neg, maxlen, vocab, tokenize_text, to_lower, thread_id = thread_id, char_level=False)

    # Appending array
    data_x = pos_data_x + neg_data_x
    data_y = pos_data_y + neg_data_y
    filename_y = pos_filename_y + neg_filename_y
    maxlen_x = max(pos_maxlen_x, neg_maxlen_x)
    lab_x = pos_lab_x + neg_lab_x

    num_hit = pos_num_hit + neg_num_hit
    unk_hit = pos_unk_hit + neg_unk_hit
    total = pos_total + neg_total

    logger.info('Thread ' + str(thread_id) + ' : ' + '  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))
    
    time_taken =  time.time() - t0
    time_taken_min = time_taken/60
    logger.info('Thread ' + str(thread_id) + ' : ' + 'Read Dataset time taken = %i sec (%.1f min)' % (time_taken, time_taken_min))

    return data_x, data_y, filename_y, maxlen_x, lab_x

def read_dataset_single(file_path, vocab, tokenize_text, to_lower, char_level=False):
    data_x = []
    indices = []
    with codecs.open(file_path, mode='r', encoding='ISO-8859-1') as input_file:
        for line in input_file:
            content = line     
            if to_lower:
                content = content.lower()
            if tokenize_text:
                content = tokenize(content)
            else:
                content = content.split()

            for word in content:
                if word in vocab:
                    indices.append(vocab[word])
                else:
                    if is_number(word):
                        indices.append(vocab['<num>'])
                    else:
                        indices.append(vocab['<unk>'])
        data_x.append(indices)

    return data_x


def get_data(args, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None):
    train_path, dev_path, test_path = args.train_path, args.dev_path, args.test_path
    vocab_size = args.vocab_size
    maxlen = 0
    
    vocab = create_vocab(train_path, maxlen, vocab_size, tokenize_text, to_lower)
    if len(vocab) < vocab_size:
        logger.warning('The vocabulary includes only %i words (less than %i)' % (len(vocab), vocab_size))
    else:
        assert vocab_size == 0 or len(vocab) == vocab_size
    logger.info('  Vocab size: %i' % (len(vocab)))

    train_x, train_y, train_filename_y, train_maxlen, train_lab_x = read_dataset(train_path, maxlen, vocab, tokenize_text, to_lower)
    dev_x, dev_y, dev_filename_y, dev_maxlen, dev_lab_x = read_dataset(dev_path, 0, vocab, tokenize_text, to_lower)
    test_x, test_y, test_filename_y, test_maxlen, test_lab_x = read_dataset(test_path, 0, vocab, tokenize_text, to_lower)

    overal_maxlen = max(train_maxlen, dev_maxlen, test_maxlen)

    return ((train_x,train_y,train_filename_y), (dev_x,dev_y,dev_filename_y), (test_x,test_y,test_filename_y), vocab, len(vocab), overal_maxlen)

# Main function wrapper that is called by main
def load_dataset(args):
    logger.info("Processing data...")

    root = ET.parse(args.train_path).getroot()

    for dialogue in root:
        for turn in dialogue:
            for sentence in turn:
                if (sentence.tag == 'speaker'):
                    logger.info(sentence.text + ':')
                if (sentence.tag == 'utterance'):
                    logger.info('  ' + sentence.text)

    # data_x is a list of lists
    ((train_x, train_y, train_filename_y),
    (dev_x, dev_y, dev_filename_y),
    (test_x, test_y, test_filename_y),
    vocab, vocab_size, overal_maxlen) = get_data(args)

    logger.info("Data processing completed!")

    with open(args.out_dir_path + '/preds/train_ref_length.txt', 'w') as instance_length_file:
        for s in train_x: instance_length_file.write('%d\n' % len(s))

    with open(args.out_dir_path + '/preds/dev_ref_length.txt', 'w') as instance_length_file:
        for s in dev_x: instance_length_file.write('%d\n' % len(s))

    with open(args.out_dir_path + '/preds/test_ref_length.txt', 'w') as instance_length_file:
        for s in test_x: instance_length_file.write('%d\n' % len(s))

    return ((train_x, train_y, train_filename_y),
        (dev_x, dev_y, dev_filename_y),
        (test_x, test_y, test_filename_y),
        vocab, vocab_size, overal_maxlen)


