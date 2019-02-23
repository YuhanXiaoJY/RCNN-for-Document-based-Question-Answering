"""
This file preprocesses the text, transforming it into the form that LineSentence can handle.
Then, you can use SGM() to get the word embeddings based on the preprocessed file and test the embeddings through testwv()
"""

from gensim.models import word2vec
import time

def testwv(word):
    modelFile = 'data/embedding/word2vec.model'
    model = word2vec.Word2Vec.load(modelFile)
    print(model.most_similar(word))

def SGM():
    time0 = time.time()
    textFile = 'data/embedding/word2vec_pre.data'
    text = word2vec.LineSentence(textFile)

    model = word2vec.Word2Vec(sentences=text, window=5, min_count=1, sg=1)
    model.save('data/embedding/word2vec.model')
    model.wv.save_word2vec_format('data/embedding/word2vec.txt', binary=False)
    time1 = time.time() - time0
    print("word2vec trained: %fs" % time1)

def predata():  # prepare data for word2vec
    output_filename = 'data/embedding/word2vec_pre.data'
    output_file = open(output_filename, 'w', encoding='utf-8')
    nameList = ['validation', 'train']
    for name in nameList:
        input_filename = 'data/preprocess_data/{}_preprocessed.data'.format(name)
        input_file = open(input_filename, 'r', encoding='utf-8')
        for line in input_file:
            line = line.strip().split('\t')
            query = line[0]
            answer = line[1]
            output_file.write('{} {}\n'.format(query, answer))
        input_file.close()
    output_file.close()


if __name__ == '__main__':
    # predata()
    # SGM()
    testwv('角色')