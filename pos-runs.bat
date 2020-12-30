@ECHO OFF
for %%h in (256) do (
	for %%b in (16) do (
		for %%c in (small) do (
			for %%v in (fasttext-alpha) do (
                for %%t in (1) do (
					for %%r in (1) do (
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
