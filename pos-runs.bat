@ECHO OFF
for %%h in (256 512) do (
	for %%b in (16) do (
		for %%c in (small mid big) do (
			for %%v in (word2vec fasttext fasttext-alpha) do (
                for %%t in (1) do (
					for %%r in (1 2 3 4 5) do (
						if exist result\pos\%%c-%%v-%%h-%%t-%%r-9 (
							echo file exists
						) else (
							echo result\pos\%%c-%%v-%%h-%%t-%%r
							python .\pos-tagger.py --corpus %%c --vector %%v -u %%h -e 15 -b %%b -s 62 -t %%t -r %%r
						)
					)
				)
			)
		)
	)
)
