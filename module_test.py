# -*- coding: utf-8 -*-
from data_util import *
import pickle
import math

def embedding_test(filename):
    embeddings = []
    word2idx = {}
    with open(filename, 'r', encoding="utf-8") as file:
        i = 0
        for line in tqdm(file, ascii=True):
            if i >= 60:
                break
            arr = line.split(" ")
            embedding = [float(val) for val in arr[1: -1]]
            word2idx[arr[0]] = i
            embeddings.append(embedding)
            i += 1

    return embeddings, word2idx

def topk_test():
    with open(FLAGS.train_sim_index, 'rb') as f:
        sim_ixs = pickle.load(f)
    i = 0
    for sim in sim_ixs:
        if i >= 32:
            break
        print(sim)
        i += 1


def words_list2index_test(words_list, word2idx,  max_len):
    """
        word list to indexes in embeddings.
        """
    unknown = word2idx.get("UNKNOWN", 0)
    num = word2idx.get("NUM", len(word2idx))
    index = [unknown] * max_len
    i = 0
    for word in words_list:
        if word in word2idx:
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

def data2index_test():
    print("[INFO]: loading embedding...")
    embedding, word2idx = load_embedding(FLAGS.embedding_file)
    questions, true_answers, false_answers, question_num = data2index(FLAGS.KB_prefile, FLAGS.train_prefile, word2idx, None,
                            FLAGS.max_sentence_len)
    batch_num = math.ceil(len(questions) / FLAGS.batch_size)
    for batch in range(batch_num):
        for i in range(batch * FLAGS.batch_size, (batch + 1) * FLAGS.batch_size):
            print(questions[i], true_answers[i], false_answers[i])

if __name__ == '__main__':
    # embeddings, word2idx = embedding_test(FLAGS.embedding_file)
    # print(word2idx)
    # print(embeddings)
    # index = words_list2index_test(['的', '将', '年', '在'],word2idx,5)
    # print(index)
    print("[INFO]: loading embedding...")
    embedding, word2idx = load_embedding(FLAGS.embedding_file)
    print('done.')
    data2index(FLAGS.KB_prefile, FLAGS.train_prefile, word2idx, None,
                            FLAGS.max_sentence_len)