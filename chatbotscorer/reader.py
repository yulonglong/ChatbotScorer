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


label_value = {
    'VALID': 1.0,
    'ACCEPTABLE': 0.5,
    'INVALID': 0.0
}

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')


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
    return tokens

def create_vocab(args, fold, x, to_lower):
    vocab_size = args.vocab_size
    logger.info('Creating vocabulary from: Fold ' + str(fold))
    total_words, unique_words = 0, 0
    word_freqs = {}

    for qna in x:
        for qora in qna:
            for word in qora:
                if to_lower:
                    word = word.lower()
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

def read_dataset(args, fold, x, y, vocab, to_lower):
    maxlen = 0
    question = x[0]
    answer = x[1]
    
    question_new = []
    answer_new = []
    
    for line in question:
        question_line_new = []
        for word in line:
            if to_lower:
                word = word.lower()
            if word in vocab:
                question_line_new.append(vocab[word])
            else:
                if is_number(word):
                    question_line_new.append(vocab['<num>'])
                else:
                    question_line_new.append(vocab['<unk>'])
        question_new.append(question_line_new)
    for line in answer:
        answer_line_new = []
        for word in line:
            if to_lower:
                word = word.lower()
            if word in vocab:
                answer_line_new.append(vocab[word])
            else:
                if is_number(word):
                    answer_line_new.append(vocab['<num>'])
                else:
                    answer_line_new.append(vocab['<unk>'])
        answer_new.append(answer_line_new)
    
    maxlen = max(maxlen, len(question_new), len(answer_new))
    data_x = [question_new, answer_new]

    data_y = []
    for labels in y:
        curr_scores = []
        for score in labels:
            curr_scores.append(label_value[score])
            
        curr_scores = np.array(curr_scores)
        if args.label_type == 'max':
            curr_scores = np.max(curr_scores)
            if curr_scores > 0 and curr_scores < 1:
                curr_scores = 1
        elif args.label_type == 'min':
            curr_scores = np.min(curr_scores)
            if curr_scores > 0 and curr_scores < 1:
                curr_scores = 0
        elif args.label_type == 'mean':
            curr_scores = np.mean(curr_scores)
        
        data_y.append(curr_scores)

    return data_x, data_y, maxlen


def process_data(args):
    x1, x2, y = [], [], []

    root = ET.parse(args.train_path).getroot()
    for dialogue in root:
        is_system = False
        curr_question = []
        curr_answer = []
        
        for turn in dialogue:
            curr_label = []
            
            for sentence in turn:
                if sentence.tag == 'speaker':
                    if sentence.text == 'SYSTEM':
                        is_system = True
                    else:
                        is_system = False
                
                elif sentence.tag == 'utterance':
                    if is_system:
                        text = tokenize(sentence.text)
                        curr_answer = text
                    else:
                        text = tokenize(sentence.text)
                        curr_question = text
                        
                elif is_system and sentence.tag == 'annotator':
                    for answer in sentence:
                        if answer.text == 'VALID' or answer.text == 'ACCEPTABLE' or answer.text == 'INVALID':
                            curr_label.append(answer.text)
            if is_system:
                # Only save data which has a label (annotated)
                if len(curr_label) > 0:
                    x1.append(curr_question)
                    x2.append(curr_answer)
                    y.append(curr_label)
                    
    x = [x1, x2]
    return x, y

def getFoldDrawCards(fold, x, y):
    train_x1, train_x2, train_y, dev_x1, dev_x2, dev_y, test_x1, test_x2, test_y = [], [], [], [], [], [], [], [], []
    validation_fold = fold+1
    if validation_fold > 9: validation_fold = 0
    for i in range(len(x[0])):
        if i%10 == fold:
            test_x1.append(x[0][i])
            test_x2.append(x[1][i])
            test_y.append(y[i])
        elif i%10 == validation_fold:
            dev_x1.append(x[0][i])
            dev_x2.append(x[1][i])
            dev_y.append(y[i])
        else:
            train_x1.append(x[0][i])
            train_x2.append(x[1][i])
            train_y.append(y[i])
            
    train_x = [train_x1, train_x2]
    dev_x = [dev_x1, dev_x2]
    test_x = [test_x1, test_x2]
    return train_x, train_y, dev_x, dev_y, test_x, test_y

# Main function wrapper that is called by main
def load_dataset(args):
    logger.info("Processing data...")

    x, y = process_data(args)
    assert len(x[0]) == len(x[1])
    assert len(x[0]) == len(y)
    logger.info('Total number of USER-SYSTEM pair : ' + str(len(y)))
    
    train_x, train_y, dev_x, dev_y, test_x, test_y = [], [], [], [], [], []
    vocab = []
    maxlen = []

    original_test_x  = []
    
    # 10-Fold cross validation, hence cut to 10
    for fold in range(10):
        curr_train_x, curr_train_y, curr_dev_x, curr_dev_y, curr_test_x, curr_test_y = getFoldDrawCards(fold, x, y)
        original_test_x.append(curr_test_x)
       
        # Process, create vocab blah blah blah here
        curr_vocab = create_vocab(args, fold, curr_train_x, to_lower=True)
        curr_train_x, curr_train_y, curr_train_maxlen = read_dataset(args, fold, curr_train_x, curr_train_y, curr_vocab, to_lower=True)
        curr_dev_x, curr_dev_y, curr_dev_maxlen = read_dataset(args, fold, curr_dev_x, curr_dev_y, curr_vocab, to_lower=True)
        curr_test_x, curr_test_y, curr_test_maxlen = read_dataset(args, fold, curr_test_x, curr_test_y, curr_vocab, to_lower=True)
        
        curr_maxlen = max(curr_train_maxlen, curr_dev_maxlen, curr_test_maxlen)
        
        train_x.append(curr_train_x)
        train_y.append(curr_train_y)
        dev_x.append(curr_dev_x)
        dev_y.append(curr_dev_y)
        test_x.append(curr_test_x)
        test_y.append(curr_test_y)
        vocab.append(curr_vocab)
        maxlen.append(curr_maxlen)

    logger.info("Data processing completed!")

    return (train_x, train_y, dev_x, dev_y, test_x, test_y, original_test_x, vocab, maxlen)
