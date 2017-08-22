import numpy as np
import cPickle as pkl
import gzip
import os
import sys

np.random.seed(1337)

output_file = 'pkl/hackabout.pkl.gz'
embeddingsPath = 'embeddings/wiki_extvec.gz'

folder = 'files/'
files = [folder+'train.txt', folder+'test.txt']

label_dict = {'Other':0, 
              'Message-Topic(e1,e2)':1, 'Message-Topic(e2,e1)':2, 
              'Product-Producer(e1,e2)':3, 'Product-Producer(e2,e1)':4, 
              'Instrument-Agency(e1,e2)':5, 'Instrument-Agency(e2,e1)':6, 
              'Entity-Destination(e1,e2)':7, 'Entity-Destination(e2,e1)':8,
              'Cause-Effect(e1,e2)':9, 'Cause-Effect(e2,e1)':10,
              'Component-Whole(e1,e2)':11, 'Component-Whole(e2,e1)':12,  
              'Entity-Origin(e1,e2)':13, 'Entity-Origin(e2,e1)':14,
              'Member-Collection(e1,e2)':15, 'Member-Collection(e2,e1)':16,
              'Content-Container(e1,e2)':17, 'Content-Container(e2,e1)':18}

words = {}
maxSentenceLen = [0,0]


def createMatrices(file, word_Ids, maxSentenceLen=100):
    labels = []
    leftContext = []
    rightContext = []

    for line in open(file):
        splits = line.strip().split('\t')
        
        label = splits[0]
        sentence = splits[1]
        tokens = sentence.split(" ")

        pos = [tokens.index('<e1s>'), tokens.index('<e1e>'), tokens.index('<e2s>'), tokens.index('<e2e>')]
        max_pos = max(pos)
        min_pos = min(pos)

        leftIds = np.zeros(maxSentenceLen+8)
        rightIds = np.zeros(maxSentenceLen+8)
        lidx = 0
        ridx = 0
        
        for idx in range(0, len(tokens)):
            if idx <= max_pos:
                leftIds[lidx+4] = getWordIdx(tokens[idx], word_Ids)
                lidx = lidx + 1
            if idx >= min_pos:
                rightIds[ridx+4] = getWordIdx(tokens[idx], word_Ids)
                ridx = ridx + 1
           
        leftContext.append(leftIds)
        rightContext.append(rightIds)
        labels.append(label_dict[label])

    return np.array(labels, dtype='int32'), np.array(leftContext, dtype='int32'), np.array(rightContext, dtype='int32'),
        
        
def getWordIdx(token, word_Ids): 
    if token in word_Ids:
        return word_Ids[token]
    elif token.lower() in word_Ids:
        return word_Ids[token.lower()]
    
    return word_Ids["UNKNOWN_TOKEN"]


for i in range(len(files)):
    file = files[i]
    for line in open(file):
        splits = line.strip().split('\t')
        label = splits[0]
        sentence = splits[1]        
        tokens = sentence.split(' ')
        pos = [tokens.index('<e1s>'), tokens.index('<e1e>'), tokens.index('<e2s>'), tokens.index('<e2e>')]
        maxSentenceLen[i] = max(maxSentenceLen[i], max(pos)+1, len(tokens)-min(pos))
        for token in tokens:
            if token not in ['<e1s>', '<e1e>', '<e2s>', '<e2e>']:
                words[token.lower()] = True
     
print("Max Sentence Lengths: ", maxSentenceLen)

word_Ids = {}
wordEmbeddings = []

fEmbeddings = gzip.open(embeddingsPath, "r")
    
print("Load pre-trained embeddings file")
for line in fEmbeddings:
    split = line.decode('utf-8').strip().split(" ")
    word = split[0]
    
    if len(word_Ids) == 0:
        word_Ids["PADDING_TOKEN"] = len(word_Ids)
        vector = np.zeros(len(split)-1)
        wordEmbeddings.append(vector)
        
        word_Ids["UNKNOWN_TOKEN"] = len(word_Ids)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)
        wordEmbeddings.append(vector)

        for w in ['<e1s>', '<e1e>', '<e2s>', '<e2e>']:
            word_Ids[w] = len(word_Ids)
            vector = np.random.uniform(-0.25, 0.25, len(split)-1)
            wordEmbeddings.append(vector)

    if word.lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word_Ids[word] = len(word_Ids)

wordEmbeddings = np.array(wordEmbeddings)

print("Embeddings shape: ", wordEmbeddings.shape)
print("Len words: ", len(words))

train_set = createMatrices(files[0], word_Ids, max(maxSentenceLen))
test_set = createMatrices(files[1], word_Ids, max(maxSentenceLen))

data = {'wordEmbeddings': wordEmbeddings, 'word_Ids': word_Ids, 
        'train_set': train_set, 'test_set': test_set}

f = gzip.open(output_file, 'wb')
pkl.dump(data, f)
f.close()

print("Data stored as " + output_file)