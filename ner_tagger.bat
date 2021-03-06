@ECHO OFF
for %%h in (256 512) do (
	for %%b in (32) do (
        for %%v in (word2vec fasttext fasttext-alpha) do (
            for %%t in (1) do (
                for %%r in (1 2 3 4 5) do (
                    if exist result\pos\%%c-%%v-%%h-%%t-%%r-9 (
                        echo file exists
                    ) else (
                        echo result\pos\%%c-%%v-%%h-%%t-%%r
                        python .\ner_tagger.py --corpus %%c  --vector %%v -u %%h -e 15 -b %%b -s 70 -t %%t -r %%r
                    )
                )
            )
        )
	)
)
