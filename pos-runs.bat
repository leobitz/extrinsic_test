@ECHO OFF
for %%h in (128) do (
	for %%b in (32) do (
		for %%c in (small mid big) do (
			for %%v in (word2vec fasttext fasttext-alpha) do (
                for %%t in (1) do (
					echo "data\result\crf-%%c-%%v-%%h-%%t.txt"
					python .\pos-tagger-crf.py --corpus %%c  --vector %%v -u %%h -e 20 -b %%b -s 62 -t %%t
				)
			)
		)
	)
)
REM python .\pos-tagger-D.py --corpus small -v word2vec -d fasttext  -u 128 -e 15 -b 32 -s 62 -t 1