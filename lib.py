

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
        # line = [w.split('/') for w in line]
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
                error +=1
                print(index, pair)
                continue
            word, tag = pair
            if word == '' :
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
            
        data.append(pairs)
    return data, list(vocab), list(tags)
