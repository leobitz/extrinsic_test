from lib import *


# words = read_words("data/clean_corpus.txt")
# vocab, word2int, int2word = build_vocab(words)
# word2freq = get_word_frequency(words, word2int, int2word)
char2int, int2char, char2tup, tup2char, n_consonant, n_vowel = build_charset()
vowels = ['e', 'u', 'i', 'a', 'ê', 'æ', 'o', 'õ', 'ø', 'ü', 'ç', 'ð']
fidel_chars = open('fidel_chars.txt', encoding='utf-8').read()

def checkIfFidel(text):
    """
    Given a text, returns true if the text contains only amharic letter or digit
    """
    text = set(text)
    for a in text:
        if a in fidel_chars:
            return True
    return False

def encode(word):
    chars = []
    if not checkIfFidel(word):
        return word
    for char in word:
        if char in fidel_chars:
            c, v = char2tup[char]
            new_v = vowels[v]
            tup = "{0}-5".format(c)
            if tup not in tup2char:
                tup = "{0}-1".format(c)
            new_c = tup2char[tup]
            if c == 5:
                chars += [new_c]
            else:
                chars += [new_c, new_v]
        else:
            chars += [char]
    return ''.join(chars)

lines = open("data/clean_corpus.txt", encoding='utf-8').read().split('\n')
abj_file = open("data/abj_clean_corpus.txt", encoding='utf-8', mode='w')
for line in lines:
    words = line.split(' ')
    abj_words = []
    for word in words:
        abj = encode(word)
        abj_words.append(abj)
    abj_line =" ".join(abj_words)
    abj_file.write(abj_line)
    abj_file.write("\n")
abj_file.close()