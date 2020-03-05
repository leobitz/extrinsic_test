import re
corpus = open('data/corpus.txt', encoding='utf-8').read().split('\n') # load the raw corpus line by line
charset = open('data/charset.txt', encoding='utf-8').read().split('\n') # load the valid character set that should be in the text
chars = [line.split(' ') for line in charset] # 
charset = []
[charset.extend(line) for line in chars]
charset.append(' ') # split the character sets into individual characters and put them in list

# Normalizing map
# character mapping that will be used to map weired (semnatically similar but different unicode chars)
# characters to common, well known semantically similar characters
char_map = {
    "(": "[ {".split(' '), # Exmaple: replace [] by (
    ")": "] }".split(' '),
    '"':"< >〈 〉 《 》 ′ ″ ‶ ‹ › ‘ ’ ‛ “ ”  ፞  ፟ ̋  ̎  ʻ ʼ ʽ ʾ ʿ » ´ « ¨ ` '".split(' '),
    "#": '0 1 2 3 4 5 6 7 8 9'.split(' '),
    "፣": '߹',
}
# character mapping that will be used to string made up of two or more characters
# that were put to signify a single character purpose
reps = {
    "።": ":: ∶∶ ：： ᎓᎓ ፡፡ ። ˸˸ ::".split(' '), # Exmaple: used for mapping ˸˸ => ።
    "፦": [':-', ': -', '፡ -', '፡-']
}
reps_map ={} # reversing the mapping
for key in reps:
    for s in reps[key]:
        reps_map[s] = key

map_char = {} # reversing the mapping
for key in char_map.keys():
    for val in char_map[key]:
        map_char[val] = key

def replace(line, map_char):
    """
    Given a text, it normazlizes by mapping unknown character to their corresponding replacement
    """
    new_chars = []
    for key in reps_map.keys():
        line = re.sub(key, reps_map[key], line) # Example: replace ': -' => '፦' from reps_map
    for char in line:
        if char in charset: # already know charatcer
            new_chars.append(char)
        else:
            if char in map_char: # character that can be mapped from the normalization made on map_char
                new_chars.append(map_char[char])
            else:
                new_chars.append(" ") # if completely unknown character, replaced with <space>
    line = " ".join("".join(new_chars).split(" "))
    return line

def clean_multiple_chars(chars, line):
    patter = re.compile('({0})'.format(chars)+"{2,}")
    line = re.sub(patter, chars, line)
    return line

space = '()!?።"፠፡፣፤፥፦፧፨%…' # characters that should have space around them
space = {c:c for c in space}
clean = open('data/clean_corpus.txt', encoding='utf-8', mode='w') # the clean file
max_word_len  = 13 # maximum word length
for line in corpus:
    line = replace(line, map_char) # normalize the line first
    line = line.strip() # remove aby trailing spaces around the text
    # if len(line) > 0: # if there are characters
    line =  clean_multiple_chars('"', line) # replace multiple '"' with single '"' e.g """" => "
    line =  clean_multiple_chars('!', line) # replace multiple '!' with single '!' e.g !!! => !
    patter = re.compile(r'(\. )'.format(chars)+"{2,}") 
    line = re.sub(patter, '…', line) # replace multiple '. ' with single '…' e.g . . . . => …
    patter = re.compile(r'(\.)'.format(chars)+"{2,}")
    line = re.sub(patter, '…', line) # replace multiple '.' with single '…' e.g ... => …
    patter = re.compile('(#)'.format(chars)+"{2,}")
    line = re.sub(patter, '#', line) # replace multiple '#' with single '#' e.g ### => #
    line = re.sub(r'\…+', "…", line) # replace multiple '…' with single '…' e.g …… => …
    line = re.sub(r'\…\s+', "…", line) # replace multiple '… ' with single '… …' e.g … … => …
    
    new_line  = []
    for c in line:
        if c in space:
            new_line.append(" {0} ".format(c)) # add space around a character if it needs
        else:
            new_line.append(c) # inset the character without adding space aorund it
    line = "".join(new_line).strip()
    line = re.sub(r'\s+', " ", line).strip() # finally, replace multiple spaces with a single one
    sp = line.split(' ')
    valid_words = []
    for s in sp: # check if every word's length is less than (max_word_len + 1)
        if len(s) <= max_word_len:
            valid_words.append(s)
    if len(valid_words) > 1: # only if the line contains more than one word, add it to the corpus
        line = " ".join(valid_words)
        clean.write(line) # write the cleaned file to the file
        clean.write("\n") # insert new line

clean.close() # close the clean file









# articles = open("raw_corpus.txt", encoding='utf-8').read().split('<,,newline,,>')
# corpus = open('corpus.txt', encoding='utf-8', mode='w')
# i = 0
# for article in articles:
#     try:
#         title, body = article.split('<,,comma,,>')
#         corpus.write(title)
#         corpus.write('\n')
#         corpus.write(body)
#         corpus.write('\n')
#     except Exception as e:
#         print(i)
   
#     i += 1
# print(i)
# corpus.close()

