# Relation Classification - Phillips Hackabout'17
Convolution neural network for relation classification between two given entities

### Step 1:
Prepare train.txt and text.txt run preprocess.py:
```
python preprocess.py
```

### Step 2:
To get the vector representation of all the words, and then save them as a pickle:
You will need the embeddings we used: https://www.cs.york.ac.uk/nlp/extvec/wiki_extvec.gz
Download the .gz file and place it in a folder called 'embeddings'
```
python prepare_data.py
```
The word embeddings, look-up table, training data and test data get saved into a pickle file in ./pkl

### Step 3:
To train the cnn model:
```
python train.py
```
All the models are saved in ./model

### Step 4:
To view the results of the above step:
```
python test.py
```
It will display the accuracy and the F1 score on the test dataset

## Dependencies:
```
NumPy
Keras
cPickle
gzip

Use python2
```
