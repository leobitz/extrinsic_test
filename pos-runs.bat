@ECHO OFF
for %%h in (128) do (
	for %%b in (32) do (
		for %%c in (small mid big) do (
			for %%v in (word2vec fasttext fasttext-alpha) do (
                for %%t in (0 1) do (
                    REM if exist "data\results\%%c-%%v-%%h-1.txt" (
                    REM     echo "file exists"
                    REM ) else (
                echo "data\result\%%c-%%v-%%h-%%t.txt"
                python .\pos-tagger.py --corpus %%c --vector %%v  -u %%h -e 15 -b %%b -s 62 -t %%t
                    REM )
				)
			)
		)
	)
)
