
for embed_size in 200
do
    for run in 5
    do
        for win_size in 1
        do
            for iter in 5
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
