import gensim
import logging
import collections
import argparse

def evaluate(vector_file, eval_file):
    model = gensim.models.KeyedVectors.load_word2vec_format(vector_file) # load the vector files on gensim model 
    model.wv.init_sims(replace=True) # init gensim internal memory
    result = model.accuracy(
        eval_file, restrict_vocab=len(model.wv.index2word), case_insensitive=False
    ) # calculate accuracy on the evaluation file
    actual_result = {} # dict for holding section
    for i in range(len(result)):
        section = result[i]["section"]
        correct = len(result[i]["correct"])
        incorrect = len(result[i]["incorrect"])
        total = correct + incorrect
        actual_result[section] = correct * 100.0 / total
    return actual_result
