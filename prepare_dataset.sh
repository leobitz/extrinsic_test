echo "Preparing all the datasets you need..."
echo "(1/4) Writing articles from raw dataset..."
python raw_to_articles.py
echo "(2/4) Cleaning articles ..."
python clean.py
echo "(3/4) Abjadization of the clean corpus ..."
python abjdizer.py --input_file "data/clean_corpus.txt" --output_file "data/abj_clean_corpus.txt"
echo "(4/4) Abjadization of the evaluation word analogy ..."
python abjdizer.py --input_file "eval/word-analogy.txt" --output_file "eval/abj-word-analogy.txt"