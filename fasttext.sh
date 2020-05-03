for embed_size in 125
do
    for run in 5
    do
        for win_size in 1
        do
            for char_size in 5
            do
                for iter in 1
                do
                    output_file="vectors/fasttext/$embed_size-$run-$win_size-$char_size-$iter-abj"
                    if [ -e $output_file ]
                    then
                        echo "File Exists"
                    else
                        # fasttext skipgram -input 'data/clean_corpus.txt' -output $output_file -dim $embed_size -epoch $run -maxn $char_size -minn 2 -ws $win_size -thread 15
                        fasttext skipgram -input 'data/abj_clean_corpus.txt' -output $output_file-abj -dim $embed_size -epoch $run -maxn $char_size -minn 2 -ws $win_size
                    fi
                done
            done
        done
    done
done