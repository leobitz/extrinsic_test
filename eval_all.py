import evaluate
import os

# load the vector file and word analogy files
# run evaluate for each element in the grid of the hyper-parameter
eval_file = "eval/word-analogy.txt"
file = open("result/fasttext.txt", mode='w')
for embed_size in [100, 200, 300]:
    for run in [4, 8, 12]:
        for win_size in [1, 3, 5]:
            for char_size in [5, 6]:
                for iter in [1]:
                    vector_file="vectors/fasttext/{0}-{1}-{2}-{3}-{4}.vec".format(embed_size, run, win_size, char_size, iter)
                    if os.path.exists(vector_file):
                        result = evaluate.evaluate(vector_file, eval_file) # evaluate
                        line = [embed_size, run, win_size, iter, result['semantic'], result['syntactic'], result['total']]
                        line = ",".join([str(x) for x in line])
                        file.write(line)
                        file.write("\n")
file.close()

# eval_file = "eval/word-analogy.txt"
# file = open("result/word2vec.txt", mode='w')
# for embed_size in [100, 200, 300]:
#     for run in [4, 8, 12]:
#         for win_size in [1, 3, 5]:
#             for iter in [1, 2, 3, 4, 5]:
#                 vector_file="vectors/word2vec/{0}-{1}-{2}-{3}".format(embed_size, run, win_size, iter)
#                 if os.path.exists(vector_file):
#                     result = evaluate.evaluate(vector_file, eval_file)
#                     line = [embed_size, run, win_size, iter, result['semantic'], result['syntactic'], result['total']]
#                     line = ",".join([str(x) for x in line])
#                     file.write(line)
#                     file.write("\n")
                    
# file.close()