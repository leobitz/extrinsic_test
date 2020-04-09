

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
