for embed_size in 100 200 300
do
    for run in 4 8 12
    do
        for win_size in 1 3 5
        do
            for char_size in 5 6
            do
                for iter in 1 2 3 4 5
                do
                    output_file="vectors/fasttext/$embed_size-$run-$win_size-$char_size-$iter"
                    if [ -e $output_file ]
                    then
                        echo "File Exists"
                    else
                        # fasttext skipgram -input 'data/clean_corpus.txt' -output $output_file -dim $embed_size -epoch $run -maxn $char_size -minn 2 -ws $win_size
                        fasttext skipgram -input 'data/abj_clean_corpus.txt' -output $output_file-abj -dim $embed_size -epoch $run -maxn $char_size -minn 2 -ws $win_size -thread 15
                    fi
                done
            done
        done
    done
done