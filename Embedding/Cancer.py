# Handle table-like data and matrices
import numpy as np
import pandas as pd

import keras
import keras.utils
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
#
from sklearn.model_selection import train_test_split , StratifiedKFold

# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt


train_variants_df = pd.read_csv("/storage/Linux Files/PycharmProjects/KaggleCancer/DataSet/training_variants")
test_variants_df = pd.read_csv("/storage/Linux Files/PycharmProjects/KaggleCancer/DataSet/test_variants")
train_text_df = pd.read_csv("/storage/Linux Files/PycharmProjects/KaggleCancer/DataSet/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
test_text_df = pd.read_csv("/storage/Linux Files/PycharmProjects/KaggleCancer/DataSet/test_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])

# Merge Variants and Text Data:
full_training = train_variants_df.merge(train_text_df, how="inner", left_on="ID", right_on="ID")
full_test = test_variants_df.merge(test_text_df, how="inner", left_on="ID", right_on="ID")

# Append Test data to Training data
full = full_training.append( full_test , ignore_index = True )
train = full[ :3321 ]

genes = pd.get_dummies( full.Gene , prefix='Gene' )
variations = pd.get_dummies( full.Variation , prefix='Variation' )


#print(text_modified)

embedded_array = np.zeros((8989,501,50), dtype=float)

text3D_array = np.load('embedded_array500.npy')
#print(text3D_array)

text500_glovness = np.mean(text3D_array, axis=1)

full_X = text3D_array
full_X = np.concatenate((genes, text3D_array) , axis=1 )
full_X = full_X.reshape((8989,501,300,1))
print("full_X shape: ", full_X.shape)

train_X = full_X[0:3321]
train_y = pd.get_dummies( train_variants_df.Class , prefix='class', prefix_sep='' )
train_y = train_y.as_matrix()
test_X = full_X[ 3321: ]
train_X , valid_X , train_y , valid_y = train_test_split( train_X , train_y , train_size = .9 )

print("train_X shape: ", train_X.shape)

layer_1 = 240
layer_2 = 240

dropout_1 = 0.5
dropout_2 = 0.5

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(10, (5, 300), activation='relu', input_shape=(501, 300, 1)))
model.add(MaxPooling2D(pool_size=(250, 1)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

history = model.fit(train_X , train_y, validation_data=(valid_X,valid_y), epochs=20, batch_size=50, initial_epoch=0)

#print (model.evaluate( train_X , train_y ))
print ("\n\nEvaluation:\n\nTraining score:", model.evaluate(train_X, train_y))
print("\nValidation score:", model.evaluate(valid_X, valid_y))
print("Layer 1: ", layer_1, "; Later 2: ", layer_2)
print("Dropout 1: ", dropout_1, "; Dropout 2: ", dropout_2)

test_Y = model.predict( test_X )
test_Y = pd.DataFrame(test_Y)
test_Y.columns = ['class1','class2','class3','class4','class5','class6','class7','class8','class9']
test_Y = pd.concat( [ test_variants_df.ID, test_Y ] , axis=1 )
test_Y.to_csv( "/storage/Linux Files/PycharmProjects/KaggleCancer/GlovePredictionsCNN.csv" , index = False )

print(history.history.keys())

# Class frequency plot
sns.set()
fig, ax = plt.subplots(nrows=2, figsize=(12,18))
# ax[0].plot(history.history['acc'], 'm')
# ax[0].plot(history.history['val_acc'], 'r')
# ax[0].set_title("Model Accuracy")
# ax[0].set_xlabel("Epochs")
# ax[0].legend(['train', 'validation'], loc='upper left')

ax[1].plot(history.history['loss'], 'k')
ax[1].plot(history.history['val_loss'] , 'b')
ax[1].set_title("Model Loss")
ax[1].set_xlabel("Epochs")
ax[1].legend(['train', 'validation' ], loc='upper left')
plt.show()


