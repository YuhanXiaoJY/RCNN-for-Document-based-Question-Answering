# -*- coding: utf-8 -*-
"""
This file preprocess the rawdata, including the raw data of train file, validation file and the test file
(you can add 'test' to the name_list in main to preprocess the test file). In my trial, I preprocessed the
knowledge base file as well. But I didn't use it in other python files(I found that the KB is so bad that it can
only worsen my result).
"""
import jieba.posseg
from tqdm import tqdm
from data_util import get_curr_dir, QAEntry


def get_stopwords(file_name):
    stopwords_list = []
    with open(file_name, encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            stopwords_list.append(line)
    return stopwords_list


def KB_preprocess(input_file_name, output_file_name):
    curr_dir = get_curr_dir()
    stopwords_filename = '{}/data/stopwords/stop_words_zh.txt'.format(curr_dir)
    stopwords_list = get_stopwords(stopwords_filename)
    input_file = open(input_file_name, 'r', encoding='utf-8')
    output_file = open(output_file_name, 'w', encoding='utf-8')

    for line in tqdm(input_file, ascii=True):
        line = line.strip()
        pairs = jieba.posseg.cut(line)
        words = [word for word, _ in pairs if word not in stopwords_list]
        entry = QAEntry(words, ' ')
        output_file.write('{}\n'.format(' '.join(entry.query)))

    input_file.close()
    output_file.close()


def data_preprocess(input_file_name, output_file_name):
    """
    data preprocess
    segment the queries and answers, add the query_id and answer_id and remove stop words
    :param input_file_name: the name of the input file
    input file format: '{}\t{}\n'.format(query, answer)
    :param output_file_name: the name of the output file
    output file format: '{}\t{}\t{}\t{}\t{}\t{}\n'.format(query, answer, query_id, answer_id, score, label)
    The query(and answer) has been segmented, saved as a list.
    :return: None
    """
    curr_dir = get_curr_dir()
    stopwords_filename = '{}/data/stopwords/stop_words_zh.txt'.format(curr_dir)
    stopwords_list = get_stopwords(stopwords_filename)
    input_file = open(input_file_name, 'r', encoding='utf-8')
    output_file = open(output_file_name, 'w', encoding='utf-8')
    prev_query = ''
    curr_query = ''
    q_id = 0
    a_id = 0


    for line in tqdm(input_file, ascii=True):
        line = line.strip().split('\t')
        curr_query = line[0]
        if curr_query == prev_query:
            a_id += 1
        else:
            q_id += 1
            a_id = 1
        line[0] = line[0].replace(' ', '')
        line[1] = line[1].replace(' ', '')  # remove the blank
        query_pairs = jieba.posseg.cut(line[0])
        query_words = [word for word, _ in query_pairs if word not in stopwords_list]
        answer_pairs = jieba.posseg.cut(line[1])
        answer_words = [word for word, _ in answer_pairs if word not in stopwords_list]

        entry = QAEntry(query_words, answer_words, label=line[2])
        entry.query_id = q_id
        entry.answer_id = a_id

        output_file.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(' '.join(entry.query), ' '.join(entry.answer),
                                                            entry.query_id, entry.answer_id, entry.score, entry.label))
        prev_query = curr_query

    input_file.close()
    output_file.close()


if __name__ == '__main__':
    curr_dir = get_curr_dir()
    name_list = ['validation', 'train']
    for name in name_list:
        input_file_name = '{}/data/rawdata/{}-set.data'.format(curr_dir, name)
        output_file_name = '{}/data/preprocess_data/{}_preprocessed.data'.format(curr_dir, name)
        data_preprocess(input_file_name, output_file_name)

    # input_file_name2 = '{}/data/KB/knowledge.txt'.format(curr_dir)
    # output_file_name2 = '{}/data/preprocess_data/knowledge_preprocessed.data'.format(curr_dir)
    # KB_preprocess(input_file_name2, output_file_name2)