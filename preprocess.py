import string

nlp = spacy.load('en')

files = ['./SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT', './SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT']
pfiles = ['./files/train.txt', './files/test.txt']

for k in range(len(files)):
	with open(files[k], 'r') as in_file:
		with open(pfiles[k], 'a') as out_file:
			data = in_file.readlines()
			for i in range(len(data)):
				splits = data[i].split('\t')
				if len(splits)==2:
					sentence = ' ' + splits[1] + ' '
					for ch in range(len(sentence)):
						if sentence[ch] == '<' and sentence[ch-1] != ' ':
							sentence = sentence[:ch] + ' ' + sentence[ch:]
						if sentence[ch] == '>' and sentence[ch+1] != ' ':
							sentence = sentence[:ch+1] + ' ' + sentence[ch+1:]

					label = data[i+1][:len(data[i+1])-2]
					tokens = sentence.strip().split()

					for j in range(len(tokens)):						
						tokens[j] = tokens[j].replace('<e1>',' <e1s> ')
						tokens[j] = tokens[j].replace('</e1>',' <e1e> ')
						tokens[j] = tokens[j].replace('<e2>',' <e2s> ')
						tokens[j] = tokens[j].replace('</e2>',' <e2e> ')

						tokens[j] = tokens[j].translate(None, string.punctuation)

						tokens[j] = tokens[j].replace('e1s','<e1s>')
						tokens[j] = tokens[j].replace('e1e','<e1e>')
						tokens[j] = tokens[j].replace('e2s','<e2s>')
						tokens[j] = tokens[j].replace('e2e','<e2e>')

					sentence = ' '.join(tokens)
					sentence = sentence.replace('    ',' ')
					sentence = sentence.replace('  ',' ')
					sentence = sentence.replace('  ',' ')
					
					final = label + '\t' + sentence + '\n'
					out_file.write(final)

	print("Succesfully created " + pfiles[k]) 