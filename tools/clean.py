import re
import string

charset = open('charset.txt', encoding='utf-8').read().split('\n') # load the valid character set that should be in the text
chars = [line.split(' ') for line in charset] # 
charset = []
[charset.extend(line) for line in chars]
charset.append(' ') # split the character sets into individual characters and put them in list
am_chars = open('am_charset.txt', encoding='utf-8').read()
fidel_chars = open('fidel_chars.txt', encoding='utf-8').read()
# Normalizing map
# character mapping that will be used to map weired (semnatically similar but different unicode chars)
# characters to common, well known semantically similar characters
char_map = {
    "(": "[ {".split(' '), # Exmaple: replace [] by (
    ")": "] }".split(' '),
    '"':"< >〈 〉 《 》 ′ ″ ‶ ‹ › ‘ ’ ‛ “ ”  ፞  ፟ ̋  ̎  ʻ ʼ ʽ ʾ ʿ » ´ « ¨ ` '".split(' '),
    "#": '0 1 2 3 4 5 6 7 8 9'.split(' '),
    "፣": '߹ :',
    " ": ' ',
    '-': '– —— —– —-'
}
# character mapping that will be used to string made up of two or more characters
# that were put to signify a single character purpose
reps = {
    "።": ":: ∶∶ ：： ᎓᎓ ፡፡ ። ˸˸ ::".split(' '), # Exmaple: used for mapping ˸˸ => ።
    "፦": [':-', ': -', '፡ -', '፡-']
}
unknown_word = "u" # unknown word to replace with unknown tokens
reps_map ={} # reversing the mapping
for key in reps:
    for s in reps[key]:
        reps_map[s] = key

map_char = {} # reversing the mapping
for key in char_map.keys():
    for val in char_map[key]:
        map_char[val] = key
puncs = list(set('()!?#.-"%…/u ' + string.punctuation))
space = '()!?።"፠፡፣፤፥፦፧፨%…' # characters that should have space around them
space = {c:c for c in space}

max_word_len  = 13 # maximum word length


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
                new_chars.append(unknown_word) # if completely unknown character, replaced with <space>
    line = " ".join("".join(new_chars).split(" "))
    return line

def clean_multiple_chars(chars, line):
    patter = re.compile('({0})'.format(chars)+"{2,}")
    line = re.sub(patter, chars, line)
    return line

def clean_series_punctuation(line, seq_len=4):
    """
    Given text, it will remove series of unknown tokens if the series has more than 5 tokens

    """
    words = line.split(" ")
    amahric = True
    changed = False
    buffer = []
    main_line = []
    for i in range(len(words)):
        if words[i] in puncs:
            if amahric:
                changed = True
            amahric = False
        else:
            if not amahric:
                changed = True
            amahric = True

        if amahric:
            # print(words[i], amahric, changed)
            if changed:
                if len(buffer) > 5:
                    buffer = [unknown_word]
                main_line.extend(buffer)
                buffer = [words[i]]
            else:
                buffer.append(words[i])
        else:
            
            if changed:
                main_line.extend(buffer)
                buffer = [words[i]]
            else:
                buffer.append(words[i])
        
        changed = False
        # print(buffer)
    if amahric:
        main_line.extend(buffer)
    else:
        if len(buffer) > 5:
            buffer = [unknown_word]
        main_line.extend(buffer)
    text =" ".join(main_line)
    return text

def replace_abbri(line):
    line = line.replace('ዶ/ ር', 'ዶ/ር')
    line = line.replace('ወ/ ሮ', 'ወ/ሮ')
    line = line.replace('ም/ ቤት', 'ም/ቤት')
    line = line.replace('ሚ/ ሩ', 'ሚ/ሩ')
    line = line.replace('ጠ/ ሚ/ ር', 'ጠ/ሚ/ር')
    line = line.replace('ዓ. ም.', 'ዓ.ም.')
    line = line.replace('ዓ. ም', 'ዓ.ም.')
    line = line.replace('መ/ ቤት', 'መ/ቤት')
    line = line.replace('ሚ/ ር', 'ሚ/ር')
    line = line.replace('ት/ ቤ', 'ት/ቤ')
    line = line.replace('መ/ ቤ', 'መ/ቤ')
    line = line.replace('ፍ/ ቤ', 'ፍ/ቤ')
    line = line.replace('ጽ/ ቤ', 'ጽ/ቤ')
    line = line.replace('ጠ/ ሚ', 'ጠ/ሚ')
    line = line.replace('ያኔ/ ኢ', 'ያኔ/ኢ')
    line = line.replace('ም/ ሚ', 'ም/ሚ')
    line = line.replace('ም/ ቤ', 'ም/ቤ')
    line = line.replace('ጄ/ ል', 'ጄ/ል')
    line = line.replace('መ/ አ', 'መ/አ')
    line = line.replace('ኰ/ ል', 'ኰ/ል')
    line = line.replace('ዓ / ም', 'ዓ.ም')
    line = line.replace('ዓ.ም. ', 'ዓ.ም ')
    line = line.replace('ፕ/ ር', 'ፕ/ር')
    line = line.replace('ጄ/ ል', 'ጄ/ል')
    line = line.replace('ከ/ ፕ', 'ከ/ፕ')
    line = line.replace('ወ/ ት', 'ወ/ት')
    line = line.replace('ቤ / ክ', 'ቤ/ክ')
    line = line.replace('ጠ / ሚ / ር', 'ጠ/ሚ/ር')
    line = line.replace('ክ/ ር', 'ክ/ር')
    line = line.replace('ፕ/ ት', 'ፕ/ት')
    line = line.replace('ፕ/ ት', 'ፕ/ት')
    line = line.replace('አ/ አ', 'አ/አ')
    line = line.replace('ጄ/ ሉ', 'ጄ/ሉ')
    
    line = line.replace('ጄ / ሉ', 'ጄ/ሉ')
    line = line.replace('አ / አ', 'አ/አ')
    line = line.replace('ፕ / ት', 'ፕ/ት')
    line = line.replace('ፕ / ት', 'ፕ/ት')
    line = line.replace('ክ / ር', 'ክ/ር')
    line = line.replace('ቤ / ክ', 'ቤ/ክ')
    line = line.replace('ወ / ት', 'ወ/ት')
    line = line.replace('ከ / ፕ', 'ከ/ፕ')
    line = line.replace('ጄ / ል', 'ጄ/ል')
    line = line.replace('ፕ / ር', 'ፕ/ር')
    line = line.replace('ዓ/ም', 'ዓ.ም.')
    line = line.replace('ኰ / ል', 'ኰ/ል')
    line = line.replace('መ / አ', 'መ/አ')
    line = line.replace('ጄ / ል', 'ጄ/ል')
    line = line.replace('ዶ / ር', 'ዶ/ር')
    line = line.replace('ወ / ሮ', 'ወ/ሮ')
    line = line.replace('ም / ቤት', 'ም/ቤት')
    line = line.replace('ሚ / ሩ', 'ሚ/ሩ')
    line = line.replace('ጠ / ሚ/ ር', 'ጠ/ሚ/ር')
    line = line.replace('ዓ. ም.', 'ዓ.ም.')
    line = line.replace('ዓ. ም', 'ዓ.ም')
    line = line.replace('መ / ቤት', 'መ/ቤት')
    line = line.replace('ሚ / ር', 'ሚ/ር')
    line = line.replace('ት / ቤ', 'ት/ቤ')
    line = line.replace('መ / ቤ', 'መ/ቤ')
    line = line.replace('ፍ / ቤ', 'ፍ/ቤ')
    line = line.replace('ጽ / ቤ', 'ጽ/ቤ')
    line = line.replace('ጠ / ሚ', 'ጠ/ሚ')
    line = line.replace('ያኔ / ኢ', 'ያኔ/ኢ')
    line = line.replace('ም / ሚ', 'ም/ሚ')
    line = line.replace('ም / ቤ', 'ም/ቤ')
    return line

def checkIfAm(text):
    """
    Given a text, returns true if the text contains any amharic letter, punctiuation or digit
    """
    text = set(text)
    for a in text:
        if a in am_chars or a in "#":
            return True
    return False

def checkIfFidel(text):
    """
    Given a text, returns true if the text contains only amharic letter or digit
    """
    text = set(text)
    for a in text:
        if a in fidel_chars or a in "#":
            return True
    return False

def replace_non_am_with_unk(line):
    """
    Given text, it replaces unknown characters with u, otherwise, it will keep a character
    """
    words = line.split(' ')
    new_words = []
    for word in words:
        if checkIfFidel(word):
            new_words.append(word)
        else:
            if len(word) > 1:
                new_words.append(unknown_word)
            else:
                new_words.append(word)
    return " ".join(new_words)

def clean_file(input_filename, output_filename):
    corpus = open(input_filename, encoding='utf-8').read() # load the raw corpus
    clean_to_file(corpus, output_filename)

def clean_to_file(corpus, output_filename):
    corpus = corpus.split('\n')
    new_corpus = open(output_filename, encoding='utf-8', mode='w') # the clean file
    for line in corpus:
        line = replace(line, map_char) # normalize the line first
        if not checkIfAm(line): # if the line doesn't contain amharic character, skip it
            continue
        line = clean_to_text(line)
        if not checkIfFidel(line): # check if at least a token contains amharic filde, if not skip the line
            continue
        line = clean_series_punctuation(line)
        new_corpus.write(line) # write the cleaned file to the file
        new_corpus.write("\n") # insert new line

    new_corpus.close() # close the clean file

def clean_to_text(line):
    line = replace_abbri(line)
    line = line.strip() # remove aby trailing spaces around the text
    # if len(line) > 0: # if there are characters
    line =  clean_multiple_chars('"', line) # replace multiple '"' with single '"' e.g """" => "
    line =  clean_multiple_chars('!', line) # replace multiple '!' with single '!' e.g !!! => !
    patter = re.compile(r'(\. )') 
    line = re.sub(patter, '…', line) # replace multiple '. ' with single '…' e.g . . . . => …
    # patter = re.compile(r'(\.)')
    # line = re.sub(patter, '…', line) # replace multiple '.' with single '…' e.g ... => …
    patter = re.compile('(#)')
    line = re.sub(patter, '#', line) # replace multiple '#' with single '#' e.g ### => #
    line = re.sub(r'#+', "#", line)
    line = re.sub('ዓ.ም.', "ዓ.ም", line)
    line = re.sub(r'(\.)\1{1,}', "…", line)
    line = re.sub(r'\…+', "…", line) # replace multiple '…' with single '…' e.g …… => …
    line = re.sub(r'\…\s+', "…", line) # replace multiple '… ' with single '… …' e.g … … => …
    line = re.sub(r'u+', "u", line) # replace multiple '… ' with single '… …' e.g … … => …
    line = re.sub(r'-+', "-", line) # replace multiple '… ' with single '- -' e.g - - => …
    line = re.sub(r'\-\s+', "- ", line) # replace multiple '… ' with single '- -' e.g - - => …
    
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
        else:
            valid_words.append(unknown_word)

    line = " ".join(valid_words)
    line = replace_non_am_with_unk(line)
    return line

# import re
# import string

# charset = open('charset.txt', encoding='utf-8').read().split('\n') # load the valid character set that should be in the text
# chars = [line.split(' ') for line in charset] # 
# charset = []
# [charset.extend(line) for line in chars]
# charset.append(' ') # split the character sets into individual characters and put them in list
# am_chars = open('am_charset.txt', encoding='utf-8').read()
# fidel_chars = open('fidel_chars.txt', encoding='utf-8').read()
# # Normalizing map
# # character mapping that will be used to map weired (semnatically similar but different unicode chars)
# # characters to common, well known semantically similar characters
# char_map = {
#     "(": "[ {".split(' '), # Exmaple: replace [] by (
#     ")": "] }".split(' '),
#     '"':"< >〈 〉 《 》 ′ ″ ‶ ‹ › ‘ ’ ‛ “ ”  ፞  ፟ ̋  ̎  ʻ ʼ ʽ ʾ ʿ » ´ « ¨ ` '".split(' '),
#     "#": '0 1 2 3 4 5 6 7 8 9'.split(' '),
#     "፣": '߹ :',
#     " ": ' ',
#     '-': '– —— —– —-'
# }
# # character mapping that will be used to string made up of two or more characters
# # that were put to signify a single character purpose
# reps = {
#     "።": ":: ∶∶ ：： ᎓᎓ ፡፡ ። ˸˸ ::".split(' '), # Exmaple: used for mapping ˸˸ => ።
#     "፦": [':-', ': -', '፡ -', '፡-']
# }
# unknown_word = "u" # unknown word to replace with unknown tokens
# reps_map ={} # reversing the mapping
# for key in reps:
#     for s in reps[key]:
#         reps_map[s] = key

# map_char = {} # reversing the mapping
# for key in char_map.keys():
#     for val in char_map[key]:
#         map_char[val] = key
# puncs = list(set('()!?#.-"%…/u ' + string.punctuation))
# space = '()!?።"፠፡፣፤፥፦፧፨%…' # characters that should have space around them
# space = {c:c for c in space}

# max_word_len  = 13 # maximum word length


# def replace(line, map_char):
#     """
#     Given a text, it normazlizes by mapping unknown character to their corresponding replacement
#     """
#     new_chars = []
#     for key in reps_map.keys():
#         line = re.sub(key, reps_map[key], line) # Example: replace ': -' => '፦' from reps_map
#     for char in line:
#         if char in charset: # already know charatcer
#             new_chars.append(char)
#         else:
#             if char in map_char: # character that can be mapped from the normalization made on map_char
#                 new_chars.append(map_char[char])
#             else:
#                 new_chars.append(unknown_word) # if completely unknown character, replaced with <space>
#     line = " ".join("".join(new_chars).split(" "))
#     return line

# def clean_multiple_chars(chars, line):
#     patter = re.compile('({0})'.format(chars)+"{2,}")
#     line = re.sub(patter, chars, line)
#     return line



# def clean_series_punctuation(line, seq_len=4):
#     """
#     Given text, it will remove series of unknown tokens if the series has more than 5 tokens

#     """
#     words = line.split(" ")
#     amahric = True
#     changed = False
#     buffer = []
#     main_line = []
#     for i in range(len(words)):
#         if words[i] in puncs:
#             if amahric:
#                 changed = True
#             amahric = False
#         else:
#             if not amahric:
#                 changed = True
#             amahric = True

#         if amahric:
#             # print(words[i], amahric, changed)
#             if changed:
#                 if len(buffer) > 5:
#                     buffer = [unknown_word]
#                 main_line.extend(buffer)
#                 buffer = [words[i]]
#             else:
#                 buffer.append(words[i])
#         else:
            
#             if changed:
#                 main_line.extend(buffer)
#                 buffer = [words[i]]
#             else:
#                 buffer.append(words[i])
        
#         changed = False
#         # print(buffer)
#     if amahric:
#         main_line.extend(buffer)
#     else:
#         if len(buffer) > 5:
#             buffer = [unknown_word]
#         main_line.extend(buffer)
#     text =" ".join(main_line)
#     return text

# def checkIfAm(text):
#     """
#     Given a text, returns true if the text contains any amharic letter, punctiuation or digit
#     """
#     text = set(text)
#     for a in text:
#         if a in am_chars:
#             return True
#     return False

# def checkIfFidel(text):
#     """
#     Given a text, returns true if the text contains only amharic letter or digit
#     """
#     text = set(text)
#     for a in text:
#         if a in fidel_chars:
#             return True
#     return False

# def replace_non_am_with_unk(line):
#     """
#     Given text, it replaces unknown characters with u, otherwise, it will keep a character
#     """
#     words = line.split(' ')
#     new_words = []
#     for word in words:
#         if checkIfFidel(word):
#             new_words.append(word)
#         else:
#             if len(word) > 1:
#                 new_words.append(unknown_word)
#             else:
#                 new_words.append(word)
#     return " ".join(new_words)

# def clean_file(input_filename, output_filename):
#     corpus = open(input_filename, encoding='utf-8').read() # load the raw corpus
#     clean_file(corpus, output_filename)

# def clean_to_file(corpus, output_filename):
#     corpus = corpus.split('\n')
#     clean = open(output_filename, encoding='utf-8', mode='w') # the clean file
#     for line in corpus:
#         if not checkIfAm(line): # if the line doesn't contain amharic character, skip it
#             continue
#         line = replace(line, map_char) # normalize the line first
#         line = line.strip() # remove aby trailing spaces around the text
#         # if len(line) > 0: # if there are characters
#         line =  clean_multiple_chars('"', line) # replace multiple '"' with single '"' e.g """" => "
#         line =  clean_multiple_chars('!', line) # replace multiple '!' with single '!' e.g !!! => !
#         patter = re.compile(r'(\. )') 
#         line = re.sub(patter, '…', line) # replace multiple '. ' with single '…' e.g . . . . => …
#         patter = re.compile(r'(\.)')
#         line = re.sub(patter, '…', line) # replace multiple '.' with single '…' e.g ... => …
#         patter = re.compile('(#)')
#         line = re.sub(patter, '#', line) # replace multiple '#' with single '#' e.g ### => #
#         line = re.sub(r'\…+', "…", line) # replace multiple '…' with single '…' e.g …… => …
#         line = re.sub(r'\…\s+', "…", line) # replace multiple '… ' with single '… …' e.g … … => …
#         line = re.sub(r'u+', "u", line) # replace multiple '… ' with single '… …' e.g … … => …
#         line = re.sub(r'-+', "-", line) # replace multiple '… ' with single '- -' e.g - - => …
#         line = re.sub(r'\-\s+', "- ", line) # replace multiple '… ' with single '- -' e.g - - => …
        
#         new_line  = []
#         for c in line:
#             if c in space:
#                 new_line.append(" {0} ".format(c)) # add space around a character if it needs
#             else:
#                 new_line.append(c) # inset the character without adding space aorund it
#         line = "".join(new_line).strip()
#         line = re.sub(r'\s+', " ", line).strip() # finally, replace multiple spaces with a single one
#         sp = line.split(' ')
#         valid_words = []
#         for s in sp: # check if every word's length is less than (max_word_len + 1)
#             if len(s) <= max_word_len:
#                 valid_words.append(s)
#             else:
#                 valid_words.append(unknown_word)
#         if len(valid_words) > 1: # only if the line contains more than one word, add it to the corpus
#             line = " ".join(valid_words)
#             line = replace_non_am_with_unk(line) # if a token doesn't contain amharic fidel, replace it with u. e.g. u#/ => U
#             if checkIfFidel(line): # check if at least a token contains amharic filde, if not skip the line
#                 line = clean_series_punctuation(line)
#                 clean.write(line) # write the cleaned file to the file
#                 clean.write("\n") # insert new line

#     clean.close() # close the clean file


# def clean_to_text(corpus):
#     corpus = corpus..split('\n')
#     cleans = []
#     for line in corpus:
#         if not checkIfAm(line): # if the line doesn't contain amharic character, skip it
#             continue
#         line = replace(line, map_char) # normalize the line first
#         line = line.strip() # remove aby trailing spaces around the text
#         # if len(line) > 0: # if there are characters
#         line =  clean_multiple_chars('"', line) # replace multiple '"' with single '"' e.g """" => "
#         line =  clean_multiple_chars('!', line) # replace multiple '!' with single '!' e.g !!! => !
#         patter = re.compile(r'(\. )') 
#         line = re.sub(patter, '…', line) # replace multiple '. ' with single '…' e.g . . . . => …
#         patter = re.compile(r'(\.)')
#         line = re.sub(patter, '…', line) # replace multiple '.' with single '…' e.g ... => …
#         patter = re.compile('(#)')
#         line = re.sub(patter, '#', line) # replace multiple '#' with single '#' e.g ### => #
#         line = re.sub(r'\…+', "…", line) # replace multiple '…' with single '…' e.g …… => …
#         line = re.sub(r'\…\s+', "…", line) # replace multiple '… ' with single '… …' e.g … … => …
#         line = re.sub(r'u+', "u", line) # replace multiple '… ' with single '… …' e.g … … => …
#         line = re.sub(r'-+', "-", line) # replace multiple '… ' with single '- -' e.g - - => …
#         line = re.sub(r'\-\s+', "- ", line) # replace multiple '… ' with single '- -' e.g - - => …
        
#         new_line  = []
#         for c in line:
#             if c in space:
#                 new_line.append(" {0} ".format(c)) # add space around a character if it needs
#             else:
#                 new_line.append(c) # inset the character without adding space aorund it
#         line = "".join(new_line).strip()
#         line = re.sub(r'\s+', " ", line).strip() # finally, replace multiple spaces with a single one
#         sp = line.split(' ')
#         valid_words = []
#         for s in sp: # check if every word's length is less than (max_word_len + 1)
#             if len(s) <= max_word_len:
#                 valid_words.append(s)
#             else:
#                 valid_words.append(unknown_word)
#         if len(valid_words) > 1: # only if the line contains more than one word, add it to the corpus
#             line = " ".join(valid_words)
#             line = replace_non_am_with_unk(line) # if a token doesn't contain amharic fidel, replace it with u. e.g. u#/ => U
#             if checkIfFidel(line): # check if at least a token contains amharic filde, if not skip the line
#                 line = clean_series_punctuation(line)
#                 cleans.append(line)
#     text = ["\n".join(cleans)]
#     return text



