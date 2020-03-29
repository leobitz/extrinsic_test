import gensim
import logging
import collections
import argparse

class Eval:

    def __init__(self, vector_file, eval_file):
        self.embed_size = 0
        self.iter = 0
        self.test_file = eval_file
        self.word2int = {}
        self.prepare(vector_file)

    def prepare(self, vector_file):
        file = open(vector_file, encoding='utf-8')
        line = file.readline().rstrip().split(' ')
        vocab_size, embed_size = int(line[0]), int(line[1])
        self.embed_size = embed_size
        vocab = []
        word2vector = {}
        k = 0
        for line in file:
            vals = line.rstrip().split(' ')
            word = vals[0]
            vec = [float(x) for x in vals[1:]]
            word2vector[word] = vec
            vocab.append(word)
            self.word2int[word] = k
            k += 1

        self.data = [vocab]
        self.model = gensim.models.Word2Vec(
            self.data, size=self.embed_size, iter=0, min_count=0)
        self.set_embeddings(self.word2int, word2vector)

    def evaluate(self):
        result = self.model.wv.accuracy(self.test_file, restrict_vocab=len(
            self.model.wv.vocab), case_insensitive=False)
        actual_result = {}
        for i in range(len(result)):
            section = result[i]['section']
            correct = len(result[i]['correct'])
            incorrect = len(result[i]['incorrect'])
            total = correct + incorrect
            actual_result[section] = correct * 100.0 / total
        return actual_result

    def set_embeddings(self, word2int, embeddings):
        """Transfers the word embedding learned bu tensorflow to gensim model
        Params:
            gensim_model - un untrained gensim_model
            word2int - dictionary that maps words to int index
            embedding - a new learned embeddings by tensorflow
        """
        self.model.wv.init_sims()
        for gindex in range(len(self.model.wv.index2word)):
            try:
                gword = self.model.wv.index2word[gindex]
                index = word2int[gword]
                embedding = embeddings[gword]
                self.model.wv.vectors_norm[gindex] = embedding
            except Exception as e:
                print(gword)

def evaluate(vector_file, eval_file):
    e = Eval(vector_file, eval_file)
    result = e.evaluate()
    return result


# parser = argparse.ArgumentParser()
# parser.add_argument('--vector_file', type=str)
# parser.add_argument('--eval_file', type=str)
# args = parser.parse_args()

# result = evaluate(args.vector_file, args.eval_file)
# print(result)