# -*- coding: utf-8 -*-
"""
This file integrates all functions related to data process.
"""
import os
from tqdm import tqdm
from gensim import corpora, models, similarities
from gensim.models import word2vec
import tensorflow as tf
import re
import sys
import random
import time
import numpy as np

# path declaration
tf.flags.DEFINE_string("KB_rawfile", "data/KB/knowledge.txt", "knowledge raw data.")
tf.flags.DEFINE_string("dictionary", "data/KB/dictionary.dict", "dictionary.")
tf.flags.DEFINE_string("corpus", "data/KB/knowledge_corpus.mm", "corpus.")
tf.flags.DEFINE_string("train_rawfile", "data/rawdata/train-set.data", "train raw data.")
tf.flags.DEFINE_string("validation_rawfile", "data/rawdata/validation-set.data", "validation raw data.")
tf.flags.DEFINE_string("train_prefile", "data/preprocess_data/train_preprocessed.data", "train preprocessed data.")
tf.flags.DEFINE_string("validation_prefile", "data/preprocess_data/validation_preprocessed.data", "validation preprocessed data.")
tf.flags.DEFINE_string("KB_prefile", "data/preprocess_data/knowledge_preprocessed.data", "knowledge preprocessed data.")
tf.flags.DEFINE_string("embedding_file", 'data/embedding/word2vec/word2vec_wx', 'word embeddings')
tf.flags.DEFINE_string('score_file', 'evaltool/sample.score.txt', 'score.file')
tf.flags.DEFINE_string('evaluate_file', 'evaltool/sample', 'evaluate')
tf.flags.DEFINE_string('result_file', 'evaltool/sample.result.txt', 'result')
tf.flags.DEFINE_string("save_file", "res/savedModel", "Save model.")

# training parameters
tf.flags.DEFINE_integer("k", 5, "K most similarity knowledge (default: 5).")
tf.flags.DEFINE_integer("rnn_size", 128, "Neurons number of hidden layer in LSTM cell (default: 100).")
tf.flags.DEFINE_float("margin", 0.1, "Constant of max-margin loss (default: 0.1).")
tf.flags.DEFINE_integer("max_grad_norm", 5, "Control gradient expansion (default: 5).")
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 50).")
tf.flags.DEFINE_integer("max_sentence_len", 40, "Maximum number of words in a sentence (default: 100).")
tf.flags.DEFINE_float("dropout_keep_prob", 0.50, "Dropout keep probability (default: 0.5).")
tf.flags.DEFINE_float("learning_rate", 0.4, "Learning rate (default: 0.4).")
tf.flags.DEFINE_float("lr_down_rate", 0.6, "Learning rate down rate(default: 0.5).")
tf.flags.DEFINE_integer("lr_down_times", 2, "Learning rate down times (default: 4)")

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 8, "Number of training epochs (default: 20)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")

FLAGS = tf.flags.FLAGS  # init flags
FLAGS(sys.argv)  # parse flags

def get_curr_dir():
    curr_dir = os.path.dirname(__file__)
    return curr_dir


class QAEntry:
    def __init__(self, query, answer, query_id=-1, answer_id=-1, score=0.0, label=-1):
        self.query = query
        self.answer = answer
        self.query_id = int(query_id)
        self.answer_id = int(answer_id)
        self.score = float(score)
        self.label = int(label)


def load_data(file_name):   # the input file must be preprocessed
    QAlist = []
    file = open(file_name, 'r', encoding='utf-8')
    if file_name.endswith('knowledge_preprocessed.data'):
        for line in tqdm(file, ascii=True):
            line = line.strip()
            text_words = line.split(' ')
            QAlist.append(QAEntry(text_words, ' '))
    else:
        for line in tqdm(file, ascii=True):
            line = line.strip().split('\t')
            query_words = line[0].split(' ')
            answer_words = line[1].split(' ')
            QAlist.append(QAEntry(query_words, answer_words, int(line[2]), int(line[3]), float(line[4]), int(line[5])))
    file.close()
    return QAlist


def load_embedding(filename):
    """
    :param filename: the filename of embedding
    :return: the list of word embeddings; the word index in the embedding list
    """
    embeddings = []
    word2idx = {}

    # load embeddings from a model file
    model = word2vec.Word2Vec.load(filename)
    i = 0
    for word in tqdm(model.wv.vocab.keys()):
        if i == 0:
            i += 1
            embeddings.append(np.zeros(256, dtype=float))
            continue
        embeddings.append(model.wv[word])
        word2idx[word] = i
        i += 1

    # # load embeddings from a txt file
    # with open(filename, 'r', encoding="utf-8") as file:
    #     i = 0
    #     for line in tqdm(file, ascii=True):
    #         if i == 0:
    #             i += 1
    #             embeddings.append(np.zeros(100, dtype=float))
    #             continue
    #         line = line.split(" ")
    #         embedding = [float(val) for val in line[1:]]
    #         word2idx[line[0]] = i
    #         embeddings.append(np.array(embedding))
    #         i += 1
    return np.array(embeddings), word2idx


def words_list2index(words_list, word2idx,  max_len):
    """
    word list to indexes in embeddings.
    """
    unknown = word2idx.get("UNKNOWN", 0)
    num = word2idx.get("NUM", len(word2idx))
    index = [unknown] * max_len
    i = 0
    for word in words_list:
        if word in word2idx.keys():
            index[i] = word2idx[word]
        else:
            if re.match("\d+", word):
                index[i] = num
            else:
                index[i] = unknown
        if i >= max_len - 1:
            break

        i += 1
    return index


def data2index(filename, word2idx, max_len):
    def val_data2index():
        val_entrys = load_data(filename)
        val_pair_num = len(val_entrys)
        val_questions, val_answers, val_labels = [], [], []
        for val_entry in val_entrys:
            val_questions.append(words_list2index(val_entry.query, word2idx, max_len))
            val_answers.append(words_list2index(val_entry.answer, word2idx, max_len))
            val_labels.append(val_entry.label)
        return val_questions, val_answers, val_labels, val_pair_num

    if filename == FLAGS.validation_prefile:
        return val_data2index()

    train_entrys = load_data(filename)

    questions, true_answers, false_answers = [], [], []
    curr_query_id = 0
    prev_query_id = 1

    queryid2true = {}
    print("[INFO]: start to get the true answers for each query id...")
    tmp1 = []
    for entry in tqdm(train_entrys):    # get the true answers for each query id
        curr_query_id = entry.query_id
        if curr_query_id != prev_query_id:
            queryid2true[prev_query_id] = tmp1
            tmp1 = []
        if entry.label == 1:
            tmp1.append(entry.answer)
            prev_query_id = curr_query_id
        else:
            prev_query_id = curr_query_id
            continue

    queryid2true[curr_query_id] = tmp1
    print("[INFO]: true answers found.")

    curr_query_id = 0
    print("[INFO]: start to get the triplets of (query, true_answer, false_answer)...")
    for entry in tqdm(train_entrys):
        curr_query_id = entry.query_id

        if entry.label == 1:
            continue
        query = []
        query.extend(entry.query)
        true_len = len(queryid2true[entry.query_id])
        if true_len > 0:
            true_answer = queryid2true[entry.query_id][random.randint(0, true_len - 1)]     # assign a random true answer
        elif true_len == 0:
            true_answer = ["UNKNOWN"]

        questions.append(words_list2index(query, word2idx, max_len))
        true_answers.append(words_list2index(true_answer, word2idx, max_len))
        false_answers.append(words_list2index(entry.answer, word2idx, max_len))
        query.clear()
    print("[INFO]: triplets finished.")
    time.sleep(0.5)

    question_num = curr_query_id
    return questions, true_answers, false_answers, question_num


if __name__ == '__main__':
    pass
    # generate_dic_and_corpus(FLAGS.KB_prefile, FLAGS.train_prefile)
    # res = topk_sim_ix(FLAGS.train_prefile, 5)
    # print(len(res))

    # curr_dir = get_curr_dir()
    # input_file_name = '{}/data/preprocess_data/preprocessed.data'.format(curr_dir)
    # QAlist = load_data(input_file_name)
    # for entry in QAlist:
    #     print('{}\t{}\t{}\t{}\t{}\t{}\n'.format(' '.join(entry.query), ' '.join(entry.answer), entry.query_id
    #                                                         , entry.answer_id, entry.score, entry.label))