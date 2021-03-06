
# the raw corpus contains, articles, their titles, comments, categories and other fields
# this script will only pick out the articles and the titles for the corpus
articles = open("data/raw_corpus.txt", encoding='utf-8').read().split('<,,newline,,>')
corpus = open('data/corpus.txt', encoding='utf-8', mode='w')
i = 0
for article in articles:
    try:
        title, body = article.split('<,,comma,,>')
        corpus.write(title)
        corpus.write('\n')
        corpus.write(body)
        corpus.write('\n')
    except Exception as e:
        print(i)
   
    i += 1
print(i)
corpus.close()
