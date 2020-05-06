@ECHO OFF
for %%h in (128) do (
	for %%b in (32) do (
		for %%c in (small mid big) do (
			for %%v in (scratch fasttext fasttext-alpha) do (
                for %%t in (0 1) do (
					for %%r in (1 2 3 4 5) do (
						if exist result\pos\%%c-%%v-%%h-%%t-%%r-9 (
							echo file exists
						) else (
							echo result\pos\%%c-%%v-%%h-%%t-%%r
							python .\pos-tagger.py --corpus %%c --vector %%v -u %%h -e 10 -b %%b -s 62 -t %%t -r %%r
						)
					)
				)
			)
		)
	)
)
