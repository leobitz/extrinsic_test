
for embed_size in 100 200 300
do
    for run in 4 8 12
    do
        for win_size in 1 3 5
        do
            for iter in 1 2 3 4 5 
            do
                output_file="vectors/word2vec/$embed_size-$run-$win_size-$iter"
                if [ -e $output_file ]
                then
                    echo "File Exists"
                else
                    word2vec -train 'data/clean_corpus.txt' -output $output_file -size $embed_size -iter $run -window $win_size -cbow 0
                fi
            done 
        done
    done
done
