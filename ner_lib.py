import numpy as np
import clean

# ዶ/ር ሥዩም ተፈራ በበኩላቸው ከከፍተኛ ትምህርት ተቋማት በአጠቃላይና


def get_words_tags(sentense, tags):
    words = []
    c = 0
    e = tags[0][0]

    portion = sentense[c:e].strip()
    portion = clean.clean_to_text(portion)
    words.extend([[w, 'O'] for w in portion.split()])
    mask = []
    for i in range(len(tags)):
        tag = tags[i]
        c = tag[0]
        e = tag[1]

        portion = sentense[c:e].strip()
        # print(portion, c, e, 'e')
        portion = clean.clean_to_text(portion)
        pl = portion.split()
        start = len(words)
        end = start + len(pl)
        words.extend([[w, tag[2]] for w in pl])
        mask.append([start, end])
        c = e
        if len(tags) > i + 1:
            e = tags[i+1][0]
        else:
            e = len(sentense)
        portion = sentense[c:e].strip()
        # print(portion, c, e, 't')
        portion = clean.clean_to_text(portion)
        words.extend([[w, 'O'] for w in portion.split()])
        c = tag[1]
    # mask = [w[1] != 'O' for w in words]
    return words, mask


def get_ner_data_v1(filename, max_len=-1):
    if max_len == -1:
        max_len = 9999
    data = []
    vocab = set()
    tagset = set()
    mask = []
    with open(filename, encoding='utf-8') as f:

        for line in f:
            sen, tags = line.split('====')
            sen = sen.strip()
            tags = tags.strip().split(',')
            tags = [tag.split('-') for tag in tags]
            tags = [[int(x[0]), int(x[1]), x[2]] for x in tags]
            pairs, m = get_words_tags(sen, tags)
            [[vocab.add(p[0]), tagset.add(p[1])] for p in pairs]
            if len(pairs) <= max_len:
                data.append(pairs)
            mask.append(m)
    return data, vocab, tagset, mask


def try_get_word_vector(word2vec, word, unk_vector):
    if '-' in word:
        word = word.replace('-', ' ')
    word = clean.clean_to_text(word)
    words = [w for w in word.split(' ') if w != '']
    v = []
    for c in words:
        if c in word2vec:
            v.append(word2vec[c])
        else:
            v.append(unk_vector)

    if len(v) > 0:
        v = np.mean(v, axis=0)
        return v
    else:
        return unk_vector


def get_word_vectors(filename, word2id, id2word):
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

    return vectors, word2vec, embed_size


def prepare_line_data(line, word2id, tag2id,
                      START_WORD_ID,
                      END_WORD_ID,
                      PAD_WORD_ID,
                      START_TAG_ID,
                      END_TAG_ID,
                      PAD_TAG_ID,
                      max_seq_length):
    x, y = [], []
    for [word, tag] in line:
        word_id = word2id[word]
        tag_id = tag2id[tag]
        x.append(word_id)
        y.append(tag_id)
    x = [START_WORD_ID] + x + [END_WORD_ID]
    y = [START_TAG_ID] + y + [END_TAG_ID]
    pad_len = max_seq_length - len(x)
    xpad = [PAD_WORD_ID] * pad_len
    ypad = [PAD_TAG_ID] * pad_len
    mask = [1]*len(x) + [0] * pad_len
    x = x + xpad
    y = y + ypad
    return x, y, mask


def get_unknown_words(train_x, test_x):
    knowns = set([])
    unknowns = set([])
    for line in train_x:
        for x in line:
            knowns.add(x)

    for line in test_x:
        for x in line:
            if x not in knowns:
                unknowns.add(x)
    return unknowns

def generate(train_x, train_y, train_m, train_t, batch_size):
    current = 0
    n_batches = len(train_x) // batch_size
    indexes = np.arange(len(train_x))
    np.random.shuffle(indexes)
    while True:
        bs = indexes[current:current+batch_size]
        x = train_x[bs]
        y = train_y[bs]
        m = train_m[bs]
        t = train_t[bs]
        yield x, y, m, t
        current += batch_size
        if current >= n_batches * batch_size:
            current = 0
            np.random.shuffle(indexes)


def evaluate_uknowns(x, preds, reals, unknowns, n_classes):
    knmat = np.zeros((n_classes, n_classes), dtype=np.float32)
    unknmat = np.zeros((n_classes, n_classes), dtype=np.float32)
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

def save_acc(accs, fold, accuracy_file):
    f = open(accuracy_file + "-" + str(fold), mode='w')
    for acc in accs:
        line = []
        for p in acc:
            line.extend(p)
        line = ",".join([str(x) for x in line])
        f.write(line)
        f.write('\n')
    f.close()