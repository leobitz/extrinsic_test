import argparse
import clean
import lib
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import ner_lib
import torch as t
import models
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

np.seterr(divide='ignore')
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--corpus", type=str)
parser.add_argument("-v", "--vector", type=str)
parser.add_argument("-u", "--units", type=int)
parser.add_argument("-e", "--epochs", type=int)
parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("-s", "--seq_length", type=int, default=62)
parser.add_argument("-t", "--train_emb", type=int, default=1)
parser.add_argument("-r", "--run", type=int, default=1)
args = parser.parse_args()

corpus = 'data/ner.txt' #+ args.corpus
if args.vector == "scratch":
    vector_file_name = None
    vec_name = "scratch"
else:
    vec_name = args.vector
    vector_file_name = 'vectors/' + args.vector + ".vec"

accuracy_file = "result/pos/{0}-{1}-{2}-{3}-{4}".format(
    args.corpus, vec_name, args.units, args.train_emb, args.run)
hidden_size = args.units
epochs = args.epochs
batch_size = args.batch_size
max_seq_len = args.seq_length
train_embedding = args.train_emb == 1

seed = 12432
random.seed(seed)
np.random.seed(seed)
t.manual_seed(seed)

test_batch_size = batch_size
k_fold = 10

start_of_sentense = '<sos>'
end_of_sentense = '<eos>'
start_of_tag = '<sot>'
end_of_tag = '<eot>'
pad_word = '<pad>'
pad_tag = '<pad>'
START_TAG = 'START'
STOP_TAG = 'STOP'

def get_ids(vocabs, tags):
    vocabs.extend([start_of_sentense, end_of_sentense, pad_word])
    tags.extend([start_of_tag, end_of_tag, pad_tag])
    # tags.extend([START_TAG, STOP_TAG])
    word2id = {word: i for i, word in enumerate(vocabs)}
    id2word = {i: word for i, word in enumerate(vocabs)}
    tag2id = {tag: tid for tid, tag in enumerate(tags)}
    id2tag = {tid: tag for tid, tag in enumerate(tags)}
    return word2id, id2word, tag2id, id2tag


data, vocab, tags, tag_spans = ner_lib.get_ner_data_v1(corpus, max_seq_len)
vocab = list(vocab)
tags = list(tags)
word2id, id2word, tag2id, id2tag = get_ids(vocab, tags)

X = []
Y = []
M = []
for line in data:
    x, y, m = ner_lib.prepare_line_data(line, word2id, tag2id,
                                        word2id[start_of_sentense],
                                        word2id[end_of_sentense],
                                        word2id[pad_word],
                                        tag2id[start_of_tag],
                                        tag2id[end_of_tag],
                                        tag2id[pad_tag],
                                        max_seq_len + 2)
    X.append(x)
    Y.append(y)
    M.append(m)
X = np.array(X)
Y = np.array(Y)
M = np.array(M).astype(np.bool)
TS2ID = np.arange(len(X))
even_len = len(X) - len(X) % k_fold
indexes = np.arange(even_len)
np.random.shuffle(indexes)
X = X[indexes]
Y = Y[indexes]
M = M[indexes]
TS2ID = TS2ID[indexes]

fold_size = even_len // k_fold
folds = {}
for i in range(k_fold):
    start = i * fold_size
    end = (i+1) * fold_size
    fold_indexes = indexes[start:end]
    fold_x = X[fold_indexes]
    fold_y = Y[fold_indexes]
    fold_m = M[fold_indexes]
    fold_T = TS2ID[fold_indexes]
    folds[i] = (fold_x, fold_y, fold_m, fold_T)

vectors = None
embed_size = 200
if vector_file_name is not None:
    vectors, word2vec, embed_size = ner_lib.get_word_vectors(vector_file_name, word2id, id2word)
    vectors[word2id[start_of_sentense]] = -np.ones(embed_size)/np.linalg.norm(-np.ones(embed_size))
    vectors[word2id[end_of_sentense]] = np.ones(embed_size)/np.linalg.norm(np.ones(embed_size))
    vectors[word2id[pad_word]] = np.zeros(embed_size)
print("Loading unks & knows")

def evaluate_uknowns(x, preds, reals, unknowns):
    knmat = np.zeros((len(tag2id), len(tag2id)), dtype=np.float32)
    unknmat = np.zeros((len(tag2id), len(tag2id)), dtype=np.float32)
    for i in range(len(x)):
        a, b = reals[i], preds[i]
        if x[i] in unknowns:
            unknmat[a, b] += 1
        else:
            knmat[a, b] += 1

    return knmat, unknmat


def test2(model, test_x, test_y, test_m, test_t, batch_size, unknowns):
    test_n_batches = len(test_x) // batch_size
    test_gen = ner_lib.generate(test_x, test_y, test_m, test_t, batch_size)
    # knmats = np.zeros((len(tags), len(tags)), dtype=np.float32)
    # unknmats = np.zeros((len(tags), len(tags)), dtype=np.float32)
    all_preds = []
    all_ys = []
    for i in range(test_n_batches):
        x, y, m, ts = next(test_gen)
        xx = t.tensor(x, dtype=t.long).cuda()
        mm = t.tensor(m, dtype=t.bool).cuda()
        z = model(xx, mm)
        z = z.detach().cpu().numpy()
        # z = t.argmax(z, dim=2).detach().cpu().numpy()
        # x, scores, z = model(xx, mm)
        for j in range(len(z)):
            tss = tag_spans[ts[j]]
            for pair in tss:
                z[j][pair[0]:pair[1]] = np.sum(z[j][pair[0]:pair[1]], axis=0)/ len(z[j][pair[0]:pair[1]])
            pred_row = z[j].argmax(axis=1)
            all_ys.extend(y[j][:k])
            all_preds.extend(pred_row[:k])
    precision = precision_score(all_ys, all_preds, average=None)
    recall = recall_score(all_ys, all_preds, average=None)
    f1 = f1_score(all_ys, all_preds, average=None)
    accuracy = accuracy_score(all_ys, all_preds)
    return np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f1)

def test(model, test_x, test_y, test_m, test_t, batch_size, unknowns):
    test_n_batches = len(test_x) // batch_size
    test_gen = ner_lib.generate(test_x, test_y, test_m, test_t, batch_size)
    knmats = np.zeros((len(tags), len(tags)), dtype=np.float32)
    unknmats = np.zeros((len(tags), len(tags)), dtype=np.float32)

    for i in range(test_n_batches):
        x, y, m, ts = next(test_gen)
        xx = t.tensor(x, dtype=t.long).cuda()
        mm = t.tensor(m, dtype=t.bool).cuda()
        z = model(xx, mm)
        z = z.detach().cpu().numpy()
        # z = t.argmax(z, dim=2).detach().cpu().numpy()
        # x, scores, z = model(xx, mm)
        for j in range(len(z)):
            tss = tag_spans[ts[j]]
            for pair in tss:
                z[j][pair[0]:pair[1]] = np.sum(z[j][pair[0]:pair[1]], axis=0)/ len(z[j][pair[0]:pair[1]])
            pred_row = z[j].argmax(axis=1)
            # pred_row = z[j]
            k = np.argwhere(y[j] == tag2id[end_of_tag])[0][0]
            knmat, unknmat = evaluate_uknowns(x[j][:k], y[j][:k], pred_row[:k], unknowns)
            knmats += knmat
            unknmats += unknmat
    return [
        ner_lib.calc_eval(knmats), 
        ner_lib.calc_eval(unknmats), 
        ner_lib.calc_eval(unknmats + knmats)
        ]

# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

def train_model(train_x, train_y, train_m,train_t, test_x, test_y, test_m, test_t, batch_size, epochs, n_batches, unknowns):
    gen = ner_lib.generate(train_x, train_y, train_m, train_t, batch_size)
    model = models.BiLSTM(len(word2id), embed_size, hidden_size, len(tag2id), vectors, train_embedding=train_embedding)
    model.apply(weights_init_uniform_rule)
    model.cuda()
    optimizer = t.optim.Adam(model.parameters(), lr=0.01)
    accs = []
    for epoch in range(epochs):
        total_loss = 0
        for batch in range(n_batches):

            x, y, m, T2S = next(gen)
            x = t.tensor(x, dtype=t.long).cuda()
            y = t.tensor(y, dtype=t.long).cuda()
            m = t.tensor(m).cuda()
            model.zero_grad()
            # x, scores, best_tag_sequence = model(x, m)
            # loss = model.loss_fn(x, m, y)
            z = model(x, m)
            z = z.view(-1, len(tag2id))
            y = y.view(-1)
            loss = model.loss_fn(z, y)
            
            loss.backward()
            optimizer.step()
            batch_loss = loss.detach().cpu().numpy()
            # print(batch_loss)
            total_loss += batch_loss

        accuracy = test2(model, test_x, test_y, test_m, test_t, test_batch_size, unknowns)
        # loss = total_loss / n_batches
        # print(print(accuracy[2]))
        # print("Epoch: {0}, Loss: {1:.3}, Test: {2:.3}, {3:.3}, {4:.3}".format(epoch, loss, accuracy[0][0], accuracy[1][0], accuracy[2][0]))
        # accuracy.insert(0, [loss])
        print(accuracy)
        
        accs.append(accuracy)
    return accs

fold_unks ={}
for i in range(k_fold):
    test_x, test_y, test_m, _ = folds[i]
    train_x, train_y, train_m = [], [], []
    for k in range(k_fold):
        if k != i:
            x, y, m, ts = folds[k]
            train_x.append(x)
    train_x = np.vstack(train_x)
    unknowns = ner_lib.get_unknown_words(train_x, test_x)
    fold_unks[i] = unknowns

for fold in range(k_fold):
    test_x, test_y, test_m, test_t = folds[fold]
    train_x, train_y, train_m, train_t = [], [], [], []
    for k in range(k_fold):
        if k != fold:
            x, y, m, ts = folds[k]
            train_x.append(x)
            train_y.append(y)
            train_m.append(m)
            train_t.extend(ts)
    train_x = np.vstack(train_x)
    train_y = np.vstack(train_y)
    train_m = np.vstack(train_m)
    train_t = np.vstack(train_t)
    print(train_t.shape)
    n_batches = len(train_x) // batch_size
    unknowns = fold_unks[fold]
    print("Fold {0}/10".format(fold))
    accss = train_model(train_x, train_y, train_m, train_t, 
                        test_x, test_y, test_m, test_t,
                        batch_size, epochs, 
                        n_batches, unknowns)
    ner_lib.save_acc(accss, fold, accuracy_file)
