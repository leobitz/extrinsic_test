# export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/bin:$LD_LIBRARY_PATH"
# export PATH="/usr/local/cuda-9.0/bin:$PATH"
# alias torch="conda activate torch"
# alias tf="conda activate tf"

# export PATH=$PATH:/home/leo/projects/amharic_word_embedding/data/fastText/build
# export PATH=$PATH:/home/leo/projects/amharic_word_embedding/word2vec

# source /home/leo/.bashrc
# cd /home/leo/projects/amharic_word_embedding/data

for embed_size in 200 300
do
    for run in 4 8 12
    do
        for win_size in 1 3 5
        do
            for iter in 1 2 3 4 5 
            do
                output_file="vectors/word2vec/$embed_size-$run-$sample-$iter"
                if [ -e output_file ]
                then
                    echo "File Exists"
                else
                    word2vec -train 'data/clean_corpus.txt' -output output_file -size $embed_size -iter $run -window $win_size -cbow 0
                fi
            done 
        done
    done
done

# /home/leo/anaconda3/envs/torch/bin/python eval.py $embed_size &