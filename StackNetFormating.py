import sklearn
import pandas as pd
import numpy as np
import xgboost as xgb

import datetime
import time

from multiprocessing import Pool
import pickle


train = pd.read_csv("./DataSet/training_variants")
test = pd.read_csv("./DataSet/test_variants")
trainx = pd.read_csv("./DataSet/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
testx = pd.read_csv("./DataSet/test_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])

train = pd.merge(train, trainx, how='left', on='ID').fillna('')
y = train['Class'].values
train = train.drop(['Class'], axis=1)

test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values

predictions = pd.read_csv("./DataSet/querry_pred.csv")
print(predictions.shape)
print(predictions)

file_pre = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')

stackNetPreds = pd.read_csv("./DataSet/querry_pred.csv")

submission = pd.DataFrame(stackNetPreds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('./result/submission_stackNet' + file_pre + '.csv', index=False)