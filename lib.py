import json
import numpy as np
import clean


def read_words(file_name):
    """
    returns list of the word sequence of the coprus
    """
    lines = open(file_name, encoding='utf-8').read().split('\n')
    words = []
    for line in lines:
        words.extend(line.split(' '))
    return words


def build_vocab(words):
    """
    builds word to integer id dictionary mapping and vice versa
    It returns sorted list of the entire vocabulary, word to id, id to word

    """
    vocab = sorted(set(words))
    word2int = {word: x for x, word in enumerate(vocab)}
    int2word = {x: word for x, word in enumerate(vocab)}
    return vocab, word2int, int2word


def get_word_frequency(words, word2int, int2word):
    """
    returns dictionary of words with thier frequencies

    """
    word2freq = {}
    for word in words:
        if word not in word2freq:
            word2freq[word] = 0
        word2freq[word] += 1.0
    return word2freq


def build_charset(charset_file="charset.txt"):
    """
    returns charcter mapping to integer and vice versa

    """
    charset = open(charset_file, encoding='utf-8').readlines()
    n_consonant = len(charset)
    n_vowel = 0
    char2int, int2char, char2tup, tup2char = {}, {}, {}, {}
    j = 0
    for k in range(len(charset)):
        row = charset[k][:-1].split(' ')
        if len(row) > n_vowel:
            n_vowel = len(row)
        for i in range(len(row)):
            char2tup[row[i]] = (k, i)
            int2char[j] = row[i]
            char2int[row[i]] = j
            tup = "{0}-{1}".format(k, i)
            tup2char[tup] = row[i]
            j += 1
    return char2int, int2char, char2tup, tup2char, n_consonant, n_vowel


def get_pos_data_v2(name, delimiter='/'):
    lines = open(name, encoding='utf-8').read().split('\n')
    data = []
    empty = 0
    legal = 0
    error = 0
    vocab = set([])
    tags = set([])
    for index, line in enumerate(lines):
        line = line.strip().split()
        split_line = []
        for pair in line:
            if '//' in pair and pair[0] == '/' and pair[1] == '/':
                word = '/'
                tag = pair.split('/')[-1]
                split_line.append([word, tag])
            else:
                split_line.append(pair.split('/'))
        pairs = []
        for pair in split_line:
            if len(pair) < 2:
                error += 1
                print(index, pair)
                continue
            if len(pair) >= 3:
                word = (' '.join(pair[:-1])).strip()
                tag = pair[-1]
                pair = [word, tag]
            word, tag = pair
            if word == '':
                empty += 1
            else:
                tag = tag.upper().strip()
                word = word.strip()
                pairs.append((word, tag))
                vocab.add(word)
                tags.add(tag)
                legal += 1
        data.append(pairs)
    # print(error, empty, legal)
    return data, list(vocab), list(tags)

# corpus = 'data/posdata/small'
# pos_data, pos_vocabs, tags = get_pos_data_v2(corpus)


def get_pos_data_v3(name, max_len=-1):
    if max_len == -1:
        max_len = 9999
    lines = open(name, encoding='utf-8').read().split('\n')
    data = []
    empty = 0
    legal = 0
    error = 0
    vocab = set([])
    tags = set([])
    for index, line in enumerate(lines):
        line = line.strip().split()
        split_line = []
        for pair in line:
            if '//' in pair and pair[0] == '/' and pair[1] == '/':
                word = '/'
                tag = pair.split('/')[-1]
                split_line.append([word, tag])
            else:
                split_line.append(pair.split('/'))
        pairs = []
        for pair in split_line:
            if len(pair) >= 3:
                word = ' '.join(pair[:-1])
                tag = pair[-1]
                pair = [word, tag]
            if len(pair) != 2:
                error += 1
                print(index, pair)
                continue
            word, tag = pair
            if word == '':
                empty += 1
            else:
                tag = tag.upper().strip()
                word = word.strip()
                # if tag == '75':
                #     print((word, tag), index)
                pairs.append((word, tag))
                vocab.add(word)
                tags.add(tag)
                legal += 1
        if len(pairs) < max_len:
            data.append(pairs)
    return data, list(vocab), list(tags)


def try_get_word_vector(word2vec, word, unk_vector):
    if '-' in word:
        word = word.replace('-', ' ')
    word = clean.clean_to_text(word)
    words = word.split(' ')
    v = []
    for c in words:
        if c in word2vec:
            v.append(word2vec[c])
        else:
            v.append(unk_vector)

    if len(v) > 0:
        v = np.sum(v, axis=0) / len(v)
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


def get_word_feats(filename):
    lines = open(filename, encoding='utf-8').read().split('\n')[:-1]
    feats = set([])
    for line in lines:
        try:
            j = json.loads(line, encoding='utf-8')
            for key in j.keys():
                if type(j[key]) == type(""):
                    # if key not in feats:
                    #     feats[key] = set(["unk"])
                    feats.add(key)
                else:
                    if key not in feats:
                        # feats[key] = {}
                        for k in j[key].keys():
                            # if k not in feats:
                                # feats[k] = set(["unk"])
                            feats.add(k)
        except Exception as e:
            pass
    feats.remove("word")
    feats.remove("pos")
    feats = list(feats)
    word2feat = {}
    for line in lines:
        my_feat = set([])
        try:
            j = json.loads(line, encoding='utf-8')
            for key in j.keys():
                if type(j[key]) == type(""):
                    my_feat.add(key)
                else:
                    if key not in feats:
                        for k in j[key].keys():
                            my_feat.add(k)
            vec = [0] * len(feats)
            my_feat.remove("word")
            if 'pos' in my_feat:
                my_feat.remove("pos")
            word = j['word']
            for elt in my_feat:
                index = feats.index(elt)
                vec[index] = 1
            word2feat[word] = vec
        except Exception as e:
            pass
    return word2feat, len(feats)
