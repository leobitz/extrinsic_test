import l3
words = list(set(open('vocab.txt', encoding='utf-8').read().split()))
# fts = l3.get_features('am')
# bin_feat = {}
# for ft in fts:
#     for f in ft:
#         if type(f) == type({}):
#             for k in f:
#                 bin_feat[k] = [0]*(len(f[k]) + 1)
    # print(ft)
# l = 0
# for k in bin_feat:
#     print(k, bin_feat[k])
#     l += len(bin_feat[k])

possible = {}
print(len(words))
proc = 0
results = []
for word in words:
    r = l3.anal_word('am', word, root=True, citation=True,  raw=True)
    l = 0
    proc += 1
    # print(word, r[0])
    s = {"word": word}
    if len(r) > 0 and len(r[0]) > 1:
        for k in r[0][1].keys():
            v = r[0][1][k]
            if hasattr(v, 'keys'):
                keys = v.keys()
                possible[k] = list(keys)
                s[k] = {}
                for kk in keys:
                    s[k][kk] = r[0][1][k][kk]
            else:
                if k in possible:
                    if v not in possible[k]:
                        possible[k].append(v)
                else:
                    possible[k] = [v]
                s[k] = v
    results.append(s)
    if proc % 100 == 0:
        print(proc * 100 / len(words))

with open('feats.txt', encoding='utf-8', mode='w') as f:
    for key in possible:
        f.write('{0} {1}\n'.format(key, " ".join([str(x) for x in possible[key]])))
with open('analysis.txt', encoding='utf-8', mode='w') as f:
    for i in range(len(results)):
        f.write(str(results[i]))
        f.write('\n')


