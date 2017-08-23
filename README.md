# HoboNet - Relation Classification
Convolution neural network for classification of semantic relationship between two given entities

**Test accuracy: 80.06%**

**F1 score (including Other): 77.45%**

**F1 score (excluding Other): 84.18%**

#### Step 1:
Prepare train.txt and text.txt run preprocess.py:
```
python preprocess.py
```

#### Step 2:
To get the vector representation of all the words, and then save them as a pickle.

You will need the embeddings we used: https://www.cs.york.ac.uk/nlp/extvec/wiki_extvec.gz

Download the .gz file and place it in a folder called 'embeddings'
```
python prepare_data.py
```
The word embeddings, look-up table, training data and test data get saved into a pickle file in ./pkl

#### Step 3:
To train the cnn model:
```
python train.py
```
All the models are saved in ./model

#### Step 4:
To view the results of the above step:
```
python test.py
```
It will display the accuracy and the F1 score on the test dataset

#### Step 5:
To use the official SemEval2010-Task8 Scorer:
```
perl scorer/semeval2010_task8_scorer-v1.2.pl results/HoboNet_result.txt scorer/test_key.txt > scorer/result_scores.txt
```
This will create result_score.txt file that contains the official scores and confusion matrix for this task

#### Dependencies:
```
NumPy
Keras
cPickle
gzip

Use python2
```

## References: 

- Pengda Qin, Weiran Xu, Jun Guo, An empirical convolutional neural network approach for semantic relation classification, Neurocomputing, Volume 190, 2016, Pages 1-9, ISSN 0925-2312, http://dx.doi.org/10.1016/j.neucom.2015.12.091
- Ngoc Thang Vu, Heike Adel, Pankaj Gupta, Hinrich Schütze, Combining Recurrent and Convolutional Neural Networks for Relation Classification, arXiv:1605.07333 [cs.CL]
- Cicero Nogueira dos Santos, Bing Xiang, Bowen Zhou, Classifying Relations by Ranking with Convolutional Neural Networks,  	arXiv:1504.06580 [cs.CL]
