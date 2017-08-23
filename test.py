import numpy as np
np.random.seed(1337)
import gzip

import sys
import cPickle as pkl

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, concatenate
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D, MaxPooling1D
from keras import regularizers

model_file = 'model/HoboNet_0.8060_0.7745.model'
result_file = 'results/HoboNet_result.txt'

batch_size = 64
nb_filter = 100
filter_length = [3, 4, 5]
hidden_dims = 100
nb_epoch = 30
reg_rate = 1e-4

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

def getPrecision(pred_test, yTest, targetLabel):
    targetLabelCount = 0
    correctTargetLabelCount = 0
    
    for idx in xrange(len(pred_test)):
        if pred_test[idx] == targetLabel:
            targetLabelCount += 1
            
            if pred_test[idx] == yTest[idx]:
                correctTargetLabelCount += 1
    
    if correctTargetLabelCount == 0:
        return 0
    
    return float(correctTargetLabelCount) / targetLabelCount


def predict_classes(prediction):
    return prediction.argmax(axis=-1)



print("Load dataset")
f = gzip.open('pkl/hackabout.pkl.gz', 'rb')
data = pkl.load(f)
f.close()

embeddings = data['wordEmbeddings']
yTrain, leftTrain, rightTrain = data['train_set']
yTest, leftTest, rightTest = data['test_set']


n_out = max(yTrain)+1

max_sentence_len = leftTrain.shape[1]


words_input_left = Input(shape=(max_sentence_len,), dtype='int32', name='words_input_left')
words_left = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)(words_input_left)
output_left3 = Convolution1D(filters=nb_filter,
                        kernel_size=filter_length[0],
                        padding='same',
                        activation='relu',
                        strides=1,
                        kernel_regularizer=regularizers.l2(reg_rate))(words_left)
output_left4 = Convolution1D(filters=nb_filter,
                        kernel_size=filter_length[1],
                        padding='same',
                        activation='relu',
                        strides=1,
                        kernel_regularizer=regularizers.l2(reg_rate))(words_left)
output_left5 = Convolution1D(filters=nb_filter,
                        kernel_size=filter_length[2],
                        padding='same',
                        activation='relu',
                        strides=1,
                        kernel_regularizer=regularizers.l2(reg_rate))(words_left)
output_left = concatenate([output_left3, output_left4, output_left5])
output_left = GlobalMaxPooling1D()(output_left)

words_input_right = Input(shape=(max_sentence_len,), dtype='int32', name='words_input_right')
words_right = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)(words_input_right)
output_right3 = Convolution1D(filters=nb_filter,
                        kernel_size=filter_length[0],
                        padding='same',
                        activation='relu',
                        strides=1,
                        kernel_regularizer=regularizers.l2(reg_rate))(words_right)
output_right4 = Convolution1D(filters=nb_filter,
                        kernel_size=filter_length[1],
                        padding='same',
                        activation='relu',
                        strides=1,
                        kernel_regularizer=regularizers.l2(reg_rate))(words_right)
output_right5 = Convolution1D(filters=nb_filter,
                        kernel_size=filter_length[2],
                        padding='same',
                        activation='relu',
                        strides=1,
                        kernel_regularizer=regularizers.l2(reg_rate))(words_right)
output_right = concatenate([output_right3, output_right4, output_right5])
output_right = GlobalMaxPooling1D()(output_right)

output = concatenate([output_left, output_right])

output = Dropout(0.5)(output)
output = Dense(n_out, activation='softmax')(output)


model = Model(inputs=[words_input_left, words_input_right], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.load_weights(model_file)

pred_test = predict_classes(model.predict([leftTest, rightTest], verbose=False))

with open(result_file, 'w') as out_file:
	for i in range(len(pred_test)):
		for key in label_dict:
			if label_dict[key] == pred_test[i]:
				out_file.write(str(i+1) + '\t' + key + '\n')

dctLabels = np.sum(pred_test)
totalDCTLabels = np.sum(yTest)

acc =  np.sum(pred_test == yTest) / float(len(yTest))
print("Test Accuracy: %.4f" % (acc))

f1Sum = 0
f1Count = 0
for targetLabel in xrange(1, max(yTest)):        
    prec = getPrecision(pred_test, yTest, targetLabel)
    rec = getPrecision(yTest, pred_test, targetLabel)
    f1 = 0 if (prec+rec) == 0 else 2*prec*rec/(prec+rec)
    f1Sum += f1
    f1Count +=1    
    
macroF1 = f1Sum / float(f1Count)    
print "Non-other Macro-Averaged F1: %.4f\n" % (macroF1)

print("\nResults stored in " + result_file)