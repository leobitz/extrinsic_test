# import numpy as np 
from lib import *
# import random

words = read_words("data/clean_corpus.txt")
vocab, word2int, int2word = build_vocab(words)
word2freq = get_word_frequency(words, word2int, int2word)
# stat = {}
# for word in word2freq:
#     val = word2freq[word]
#     if val not in stat:
#         stat[val] = 0
#     stat[val] += 1
# for i in range(10):
#     if i in stat:
#         print("{0} : {1}".format(i, stat[i]))

vec_size = 50
vec_file = open("vectors/random/vec.txt", encoding='utf-8', mode='w')
vec_file.write("{0} {1}\n".format(len(vocab), vec_size))
# vec = np.random.rand(vec_size)
k = 0
for word in vocab:
    if word2freq[word] > 5:
        line = "{0} {1}\n".format(word, " ".join([".5434" for x in range(vec_size)]))
        vec_file.write(line)
        k += 1
print("after mini-threshold: " + str(k))
vec_file.close()
