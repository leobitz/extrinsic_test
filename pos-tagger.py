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
# from crf.crf import ConditionalRandomField
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


np.seterr(divide='ignore')
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--corpus", type=str)
parser.add_argument("-v", "--vector", type=str)
# parser.add_argument("-r", "--result", type=str)
parser.add_argument("-u", "--units", type=int)
parser.add_argument("-e", "--epochs", type=int)
parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("-s", "--seq_length", type=int, default=62)
parser.add_argument("-t", "--train_emb", type=int, default=1)
args = parser.parse_args()

corpus = 'data/posdata/' + args.corpus
if args.vector is None:
    vector_file_name = None
    vec_name = "scratch"
else:
    vec_name = args.vector
    vector_file_name = 'vectors/' + args.vector + ".vec"

accuracy_file = "result/pos/{0}-{1}-{2}-{3}".format(args.corpus, vec_name, args.units, args.train_emb)
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

train_ratio = .7
test_batch_size = batch_size
k_fold = 10

pos_data, pos_vocabs, tags = lib.get_pos_data_v2(corpus)
p = []
for line in pos_data:
    if len(line) < max_seq_len:
        p.append(line)
pos_data = p

clean_data, clean_pos_vocabs, total_words = pos_data, pos_vocabs, tags
clean_pos_vocabs.extend([start_of_sentense, end_of_sentense, pad_word])
tags.extend([start_of_tag, end_of_tag, pad_tag])
word2id = {word: i for i, word in enumerate(clean_pos_vocabs)}
id2word = {i: word for i, word in enumerate(clean_pos_vocabs)}
tag2id = {tag: tid for tid, tag in enumerate(tags)}
id2tag = {tid: tag for tid, tag in enumerate(tags)}



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
    return vectors, word2vec, embed_size

def prepare_line_data(line,  max_seq_length):
    x, y = [], []
    for [word, tag] in line:
        word_id = word2id[word]
        tag_id = tag2id[tag]
        x.append(word_id)
        y.append(tag_id)
    x = [word2id[start_of_sentense]] + x + [word2id[end_of_sentense]]
    y = [tag2id[start_of_tag]] + y + [tag2id[end_of_tag]]
    pad_len = max_seq_length - len(x)
    xpad = [word2id[pad_word]] * pad_len
    ypad = [tag2id[pad_tag]] * pad_len
    mask = [1]*len(x) + [0] * pad_len
    x = x + xpad
    y = y + ypad
    return x, y, mask


X = []
Y = []
M = []
for line in clean_data:
    x, y, m = prepare_line_data(line, max_seq_len + 2)
    X.append(x)
    Y.append(y)
    M.append(m)
X = np.array(X)
Y = np.array(Y)
M = np.array(M)
even_len = len(X) - len(X) % k_fold
indexes = np.arange(even_len)
np.random.shuffle(indexes)
X = X[indexes]
Y = Y[indexes]
M = M[indexes]

fold_size = even_len // k_fold
folds = {}
for i in range(k_fold):
    start = i * fold_size
    end = (i+1) * fold_size
    fold_indexes = indexes[start:end]
    fold_x = X[fold_indexes]
    fold_y = Y[fold_indexes]
    fold_m = M[fold_indexes]
    folds[i] = (fold_x, fold_y, fold_m)

vectors = None
embed_size = 125
if vector_file_name is not None:
    vectors, word2vec, embed_size = get_word_vectors(vector_file_name)
print("Loading unks & knows")


def get_unknown_words(train_x, train_y):
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


def generate(train_x, train_y, train_m, batch_size):
    current = 0
    n_batches = len(train_x) // batch_size
    indexes = np.arange(len(train_x))
    np.random.shuffle(indexes)
    while True:
        bs = indexes[current:current+batch_size]
        x = train_x[bs]
        y = train_y[bs]
        m = train_m[bs]
        yield x, y, m
        current += batch_size
        if current >= n_batches * batch_size:
            current = 0
            np.random.shuffle(indexes)


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


def calc_eval(confusion):
    total = confusion.sum()
    TPs = np.array([confusion[x, x] for x in range(len(confusion))])
    FPs = np.array([sum(confusion[:, x]) - confusion[x, x]
                    for x in range(len(confusion))])
    FNs = np.array([sum(confusion[x, :]) - confusion[x, x]
                    for x in range(len(confusion))])
    TNs = np.array([sum(confusion[x, :]) - confusion[x, x]
                    for x in range(len(confusion))])
    P = np.nan_to_num([TPs[x]/(TPs[x] + FPs[x]) for x in range(len(TPs))])
    R = np.nan_to_num([TPs[x]/(TPs[x] + FNs[x]) for x in range(len(TPs))])
    A = sum(TPs) / total
    F1 = np.nan_to_num(2 * np.mean(P) * np.mean(R) / (np.mean(P) + np.mean(R)))
    return A, np.mean(P), np.mean(R), F1


def test(model, test_x, test_y, test_m, batch_size, unknowns):
    test_n_batches = len(test_x) // batch_size
    test_gen = generate(test_x, test_y, test_m, batch_size)
    knmats = np.zeros((len(tags), len(tags)), dtype=np.float32)
    unknmats = np.zeros((len(tags), len(tags)), dtype=np.float32)
    all_preds = []
    all_targets = []
    for i in range(test_n_batches):
        x, y, m = next(test_gen)
        xx = t.tensor(x, dtype=t.long).cuda()
        mm = t.tensor(m, dtype=t.long).cuda()
        z = model(xx, mm)
        preds = t.argmax(z, dim=2).detach().cpu().numpy()
        for j in range(len(preds)):
            k = np.argwhere(y[j] == tag2id[end_of_tag])[0][0]
            knmat, unknmat = evaluate_uknowns(x[j][:k], y[j][:k], preds[j][:k], unknowns)
            knmats += knmat
            unknmats += unknmat
    # return acc
    return calc_eval(knmats), calc_eval(unknmats), calc_eval(unknmats + knmats)


class Tagger(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, n_classes, embedding_vectors=None, train_embedding=True):
        super(Tagger, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embedding_vectors is not None:
            self.embedding.weight.data.copy_(t.from_numpy(embedding_vectors))
            self.embedding.weight.requires_grad = train_embedding

        self.lstm = nn.LSTM(embed_size, hidden_size//2, #num_layers=2, dropout=.2,
                            batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size, n_classes)

    def forward(self, x, mask,  y=None):
        x = self.embedding(x)
        x, (h, c) = self.lstm(x)
        x = self.fc1(x)
        return x

    def loss_fn(self, logits, target, mask):
        log_likelihood = self.crf(logits, target, mask)
        return -log_likelihood / logits.shape[0]

def train_model(train_x, train_y, train_m, test_x, test_y, test_m, batch_size, epochs, n_batches, unknowns):
    gen = generate(train_x, train_y, train_m, batch_size)
    model = Tagger(len(word2id), embed_size, hidden_size, len(tag2id), vectors, train_embedding=train_embedding)
    model.cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=0.001)
    accs = []
    for epoch in range(epochs):
        total_loss = 0
        for batch in range(n_batches):

            x, y, m = next(gen)
            x = t.tensor(x, dtype=t.long).cuda()
            y = t.tensor(y, dtype=t.long).cuda()
            m = t.tensor(m, dtype=t.long).cuda()
            model.zero_grad()
            z = model(x, m, y)
            z = z.view(-1, len(tag2id))
            y = y.view(-1)
            loss = loss_function(z, y)
            loss.backward()
            optimizer.step()

            batch_loss = loss.detach().cpu().numpy()
            total_loss += batch_loss
        accuracy = test(model, test_x, test_y, test_m, test_batch_size, unknowns)
        accs.append(accuracy)
        print(epoch, total_loss / n_batches, accuracy)
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
    test_x, test_y, test_m = folds[i]
    train_x, train_y, train_m = [], [], []
    for k in range(k_fold):
        if k != i:
            x, y, m = folds[k]
            train_x.append(x)
    train_x = np.vstack(train_x)
    knowns, unknowns, test_knowns, unk_mask = get_unknown_words(train_x, test_x)
    fold_unks[i] = (knowns, unknowns, test_knowns, unk_mask)

print("Starting training")

for fold in range(k_fold):
    test_x, test_y, test_m = folds[fold]
    train_x, train_y, train_m = [], [], []
    for k in range(k_fold):
        if k != fold:
            x, y, m = folds[k]
            train_x.append(x)
            train_y.append(y)
            train_m.append(m)
    train_x = np.vstack(train_x)
    train_y = np.vstack(train_y)
    train_m = np.vstack(train_m)
    n_batches = len(train_x) // batch_size
    (knowns, unknowns, test_knowns, mask) = fold_unks[fold]
    print("Fold {0}/10".format(fold))
    accss = train_model(train_x, train_y, train_m, test_x, test_y, test_m, batch_size, epochs, n_batches, unknowns)
    save_acc(accss, fold)
