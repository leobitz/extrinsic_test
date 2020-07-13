import argparse
import clean
import lib
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch as t
import os
import models
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

corpus = 'data/posdata/' + args.corpus
if args.vector == "scratch":
    vector_file_name = None
    vec_name = "scratch"
else:
    vec_name = args.vector
    vector_file_name = 'vectors/' + args.vector + ".vec"

accuracy_file = "result/pos/{0}-{1}-{2}-{3}-{4}".format(args.corpus, vec_name, args.units, args.train_emb, args.run)
hidden_size = args.units
epochs = args.epochs
batch_size = args.batch_size
max_seq_len = args.seq_length
train_embedding = args.train_emb == 1

seed = 12432
random.seed(seed)
np.random.seed(seed)
t.manual_seed(seed)

start_of_sentense = '<sos>'
end_of_sentense = '<eos>'
start_of_tag = '<sot>'
end_of_tag = '<eot>'
pad_word = '<pad>'
pad_tag = '<pad>'
pad_char = 'k'
unk_char = 'n'

train_ratio = .7
test_batch_size = batch_size
k_fold = 10
max_char_length = 14

pos_data, pos_vocabs, tags = lib.get_pos_data_v2(corpus)
p = []
for line in pos_data:
    if len(line) < max_seq_len:
        p.append(line)
pos_data = p

charset = list(open('data/charset.txt', encoding='utf-8').read()) + [pad_char, unk_char]
charset.pop(charset.index(' '))
charset.pop(charset.index('\n'))
# charset.pop(charset.index(''))
clean_data, clean_pos_vocabs, total_words = pos_data, pos_vocabs, tags
clean_pos_vocabs.extend([start_of_sentense, end_of_sentense, pad_word])
tags.extend([start_of_tag, end_of_tag, pad_tag])
word2id = {word: i for i, word in enumerate(clean_pos_vocabs)}
id2word = {i: word for i, word in enumerate(clean_pos_vocabs)}
tag2id = {tag: tid for tid, tag in enumerate(tags)}
id2tag = {tid: tag for tid, tag in enumerate(tags)}
char2id = {c:i for i, c in enumerate(charset)}
id2char = {i:c for i, c in enumerate(charset)}


def try_get_word_vector(allw2v, word, unk_vector):
    if '-' in word:
        word = word.replace('-', ' ')
    word = clean.clean_to_text(word)
    words = word.split(' ')
    v = []
    for c in words:
        if c in allw2v:
            v.append(allw2v[c])
        else:
            v.append(unk_vector)

    if len(v) > 0:
        v = np.sum(v, axis=0) / len(v)
        return v
    else:
        return unk_vector

def get_word_vectors(filename):
    word2vec = {}
    allw2v = {}
    with open(filename, encoding='utf-8') as f:
        line = f.readline().strip().split()
        vocab_size, embed_size = int(line[0]), int(line[1])
        for line in f:
            line = line.strip().split()
            word, vec = line[0], [float(x) for x in line[1:]]
            word = word.strip()
            allw2v[word] = np.array(vec)
    unk_vector = allw2v['u']


    for word in word2id.keys():
        if word in allw2v:
            word2vec[word] = allw2v[word]
        else:
            word2vec[word] = try_get_word_vector(allw2v, word, unk_vector)

    vectors = np.empty((len(id2word), embed_size))
    for word_id in id2word.keys():
        vec = word2vec[id2word[word_id]]
        vec = np.array(vec)
        if np.count_nonzero(vec) > 0:
            norm = np.linalg.norm(vec)
            vec = vec / norm
        vectors[word_id] = vec

    vectors[word2id[start_of_sentense]] = -np.ones(embed_size)/np.linalg.norm(-np.ones(embed_size))
    vectors[word2id[end_of_sentense]] = np.ones(embed_size)/np.linalg.norm(np.ones(embed_size))
    vectors[word2id[pad_word]] = np.zeros(embed_size)
    return vectors, word2vec, embed_size, unk_vector

def word_to_ids(word, char2id, max_len=13):
    idx = -1
    chars = []
    for c in word:
        if c in char2id:
            idx = char2id[c]
        else:
            idx = char2id[unk_char]
        chars.append(idx)
    chars = chars + [char2id[pad_char]] * (max_len - len(chars))
    return chars

def prepare_line_data(line,  max_seq_length, max_char_length):
    x, y = [], []
    char_x = []
    for [word, tag] in line:
        word_id = word2id[word]
        tag_id = tag2id[tag]
        x.append(word_id)
        y.append(tag_id)
        char_x.append(word_to_ids(word, char2id, max_char_length))
    x = [word2id[start_of_sentense]] + x + [word2id[end_of_sentense]]
    y = [tag2id[start_of_tag]] + y + [tag2id[end_of_tag]]
    pad_char_word = [char2id[pad_char]]*max_char_length
    char_x = [pad_char_word] + char_x + [pad_char_word]
    pad_len = max_seq_length - len(x)
    xpad = [word2id[pad_word]] * pad_len
    ypad = [tag2id[pad_tag]] * pad_len
    c_pad = [pad_char_word] * pad_len
    mask = [1]*len(x) + [0] * pad_len
    x = x + xpad
    y = y + ypad
    c = char_x + c_pad
    return x, y, c, mask


X = []
Y = []
M = []
C = []
for line in clean_data:
    x, y, c, m = prepare_line_data(line, max_seq_len + 2, max_char_length)
    X.append(x)
    Y.append(y)
    C.append(c)
    M.append(m)
X = np.array(X)
Y = np.array(Y)
M = np.array(M)
C = np.array(C)
even_len = len(X) - len(X) % k_fold
indexes = np.arange(even_len)
np.random.shuffle(indexes)
X = X[indexes]
Y = Y[indexes]
M = M[indexes]
C = C[indexes]

fold_size = even_len // k_fold
folds = {}
for i in range(k_fold):
    start = i * fold_size
    end = (i+1) * fold_size
    fold_indexes = indexes[start:end]
    fold_x = X[fold_indexes]
    fold_y = Y[fold_indexes]
    fold_m = M[fold_indexes]
    fold_c = C[fold_indexes]
    folds[i] = (fold_x, fold_y, fold_c, fold_m)

vectors = None
embed_size = 200
if vector_file_name is not None:
    vectors, word2vec, embed_size, unk_vector = get_word_vectors(vector_file_name)
print("Loading unks & knows")


def get_unknown_words(train_x, test_x):
    knowns = set([])
    unknowns = set([])
    test_knowns = set([])
    for line in train_x:
        for x in line:
            knowns.add(x)
    mat = []
    for line in test_x:
        row = []
        for x in line:
            if x not in knowns:
                unknowns.add(x)
                row.append(1)
            else:
                test_knowns.add(x)
                row.append(0)
        mat.append(row)
    mat = np.array(mat, np.float32)
    return knowns, unknowns, test_knowns, mat


def generate(train_x, train_y, train_c,  train_m, batch_size):
    current = 0
    n_batches = len(train_x) // batch_size
    indexes = np.arange(len(train_x))
    np.random.shuffle(indexes)
    while True:
        bs = indexes[current:current+batch_size]
        x = train_x[bs]
        y = train_y[bs]
        m = train_m[bs]
        c = train_c[bs]
        yield x, y, c, m
        current += batch_size
        if current >= n_batches * batch_size:
            current = 0
            np.random.shuffle(indexes)

def evaluate(targtes, preds):
    all_ys = targtes
    all_preds = preds
    precision = precision_score(all_ys, all_preds, average=None)
    recall = recall_score(all_ys, all_preds, average=None)
    f1 = f1_score(all_ys, all_preds, average=None)
    accuracy = accuracy_score(all_ys, all_preds)
    return np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f1)

def test_model(model, test_data, batch_size, unknowns):
    test_n_batches = len(test_data[0]) // batch_size
    test_gen = generate(test_data[0], test_data[1], test_data[2], test_data[3], batch_size)
    knmats = np.zeros((len(tags), len(tags)), dtype=np.float32)
    unknmats = np.zeros((len(tags), len(tags)), dtype=np.float32)
    all_preds = []
    all_ys = []
    allo = []
    unks = []
    for i in range(test_n_batches):
        x, y, c, m = next(test_gen)
        xx = t.tensor(x, dtype=t.long).cuda()
        mm = t.tensor(m, dtype=t.long).cuda()
        c = t.tensor(c, dtype=t.long).cuda()
        z = model(xx, c)
        preds = t.argmax(z, dim=2).detach().cpu().numpy()
        for j in range(len(preds)):
            k = np.argwhere(y[j] == tag2id[end_of_tag])[0][0]
            pred_row = preds[j]
            for uk  in range(k):
                py = y[j][uk]
                pp = pred_row[uk]
                px = x[j][uk]
                # print([px, py, pp])
                if px in unknowns:
                    allo.append([px, py, pp])
                else:
                    unks.append([px, py, pp])
    allo = np.array(allo) 
    unks = np.array(unks)
    every = np.concatenate((allo, unks), axis=0)
    ek = evaluate(allo[:, 1], allo[:, 2])
    eu = evaluate(unks[:, 1], unks[:, 2])
    ee = evaluate(every[:, 1], every[:, 2])
    return [ek, eu, ee]

def train_model(train, test_data, batch_size, epochs, n_batches, unknowns):
    gen = generate(train[0], train[1], train[2], train[3], batch_size)
    model = models.BiLSTMChar(len(word2id), embed_size, hidden_size, len(tag2id), len(char2id), 32, max_char_length, vectors, train_embedding=train_embedding)
    model.cuda()
    optimizer = t.optim.Adam(model.parameters(), lr=0.001)
    accs = []
    for epoch in range(epochs):
        total_loss = 0
        for batch in range(n_batches):

            x, y, c, m = next(gen)
            x = t.tensor(x, dtype=t.long).cuda()
            y = t.tensor(y, dtype=t.long).cuda()
            m = t.tensor(m, dtype=t.long).cuda()
            c = t.tensor(c, dtype=t.long).cuda()
            model.zero_grad()
            z = model(x, c)
            z = z.view(-1, len(tag2id))
            y = y.view(-1)
            loss = model.loss_func(z, y)
            loss.backward()
            optimizer.step()

            batch_loss = loss.detach().cpu().numpy()
            total_loss += batch_loss
        accuracy = test_model(model, test_data, test_batch_size, unknowns)
        accs.append(accuracy)
        loss = total_loss / n_batches
        print("Epoch: {0}, Loss: {1:.3}, Test: {2:.3}, {3:.3}, {4:.3}".format(epoch, loss, accuracy[0][-1], accuracy[1][-1], accuracy[2][-1]))
        accuracy.insert(0, [loss])
    return accs


def save_acc(accs, fold):
    f = open(accuracy_file + "-" + str(fold), mode='w')
    for acc in accs:
        line = []
        for p in acc:
            line.extend(p)
        line = ",".join([str(x) for x in line])
        f.write(line)
        f.write('\n')
    f.close()



fold_unks ={}
for i in range(k_fold):
    test_x, _, _, _ = folds[i]
    train_x = []
    for k in range(k_fold):
        if k != i:
            x, y, c, m = folds[k]
            train_x.append(x)
    train_x = np.vstack(train_x)
    knowns, unknowns, test_knowns, unk_mask = get_unknown_words(train_x, test_x)
    fold_unks[i] = (knowns, unknowns, test_knowns, unk_mask)

print("Starting training")

for fold in range(k_fold):
    test_x, test_y, test_c, test_m = folds[fold]
    train_x, train_y, train_c, train_m = [], [], [], []
    for k in range(k_fold):
        if k != fold:
            x, y, c, m = folds[k]
            train_x.append(x)
            train_y.append(y)
            train_m.append(m)
            train_c.append(c)
    train_x = np.vstack(train_x)
    train_y = np.vstack(train_y)
    train_c = np.vstack(train_c)
    train_m = np.vstack(train_m)
    n_batches = len(train_x) // batch_size
    (knowns, unknowns, test_knowns, mask) = fold_unks[fold]
    print("Fold {0}/10".format(fold))
    train = (train_x, train_y, train_c, train_m)
    test_data = (test_x, test_y, test_c, test_m)
    accss = train_model(train, test_data, batch_size, epochs, n_batches, unknowns)
    save_acc(accss, fold)
