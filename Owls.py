from sklearn import preprocessing, pipeline, feature_extraction, decomposition, model_selection, metrics, svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import normalize, Imputer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

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


print(print row['text'][0])
# df_all = pd.concat((train, test), axis=0, ignore_index=True)
# df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
# df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)
#
# print(df_all)
#
# gen_var_lst = sorted(list(train.Gene.unique()) + list(train.Variation.unique()))
# print(len(gen_var_lst))
#
# gen_var_lst = [x for x in gen_var_lst if len(x.split(' '))==1]
# print(len(gen_var_lst))
# i_ = 0
#
# #commented for Kaggle Limits
# for gen_var_lst_itm in gen_var_lst:
#     if i_ % 100 == 0: print(i_)
#     df_all['GV_'+str(gen_var_lst_itm)] = df_all['Text'].map(lambda x: str(x).count(str(gen_var_lst_itm)))
#     i_ += 1
#
# for c in df_all.columns:
#     if df_all[c].dtype == 'object':
#         if c in ['Gene','Variation']:
#             lbl = preprocessing.LabelEncoder()
#             df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)
#             df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
#             df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
#         elif c != 'Text':
#             lbl = preprocessing.LabelEncoder()
#             df_all[c] = lbl.fit_transform(df_all[c].values)
#         if c=='Text':
#             df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
#             df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
#
# df_all.to_csv( "./DataSet/DataframeAll.csv" , index = False )

df_all = pd.read_csv("./DataSet/DataframeAll.csv")

train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]

class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x.drop(['Gene', 'Variation','ID','Text'],axis=1).values
        return x

class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.key].apply(str)

#commented for Kaggle Limits
# ('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')),
#                            ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))),
#                            ('tsvd3', decomposition.TruncatedSVD(n_components=50, n_iter=25, random_state=12))]))

print('Pipeline...')
fp = pipeline.Pipeline([
    ('union', pipeline.FeatureUnion(
        n_jobs=-1,
        transformer_list=[
            ('standard', cust_regression_vals()),
            ('pi1', pipeline.Pipeline([('Gene', cust_txt_col('Gene')),
                                       ('count_Gene',
                                        feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))),
                                       ('tsvd1',
                                        decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            ('pi2', pipeline.Pipeline([('Variation', cust_txt_col('Variation')),
                                       ('count_Variation',
                                        feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))),
                                       ('tsvd2',
                                        decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),

            ('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')),
                                        ('hv', feature_extraction.text.HashingVectorizer(decode_error='ignore', n_features=2 ** 16, non_negative=True, ngram_range=(1, 3))),
                                        ('tfidf_Text', feature_extraction.text.TfidfTransformer()),
                                        ('tsvd3', decomposition.TruncatedSVD(n_components=300, n_iter=25, random_state=12))]))

        ])
     )])


# t = time.time()
# train = fp.fit_transform(train)
# elapsed = time.time() - t
# print("Train pipeline time: ", elapsed)
#
# np.save("./DataSet/trainEngineered", train)


# filename = './result/finalized_pipeline.sav'
# #pickle.dump(fp, open(filename, 'wb'))
#
# fp = pickle.load(open(filename, 'rb'))
#
y = y - 1 #fix for zero bound array
y = pd.DataFrame(y, index=None, columns=None)
train = np.load("./DataSet/trainEngineered.npy")
trainCSV = pd.DataFrame(train, index=None, columns=None)
trainCSV = pd.concat((y,trainCSV), axis=1)
#trainCSV.columns = range(trainCSV.shape[1])
trainCSV.to_csv("./DataSet/trainEngineered.csv",index=False)
#print(trainCSV.shape)
#print(trainCSV)
predictions = pd.read_csv("./DataSet/querry_pred.csv", header=None)
print(predictions.shape)
print(predictions)
predictions = pd.DataFrame((predictions.values), columns=['class'+str(c+1) for c in range(9)])
predictions = pd.concat( [ testx.ID, predictions ] , axis=1 )
print(predictions.shape)
print(predictions)
file_pre = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
predictions.to_csv('./result/submission_stacknet_' + file_pre + '.csv', index=False)

#
# print(train.shape)
# print("========================")
# print("========================")
# print(train)
# print("========================")
# print("========================")
#
#
# t = time.time()
# print('Test Transform Pipeline...')
# test = fp.transform(test)
# elapsed = time.time()-t
# print("Test pipeline time: ", elapsed)
# print(test.shape)

# test_t = np.empty([0, train.shape[1]])
# step = 200
# print(test_t.shape)
#
#
#
# for i in range(0, len(test), step):
#     step_end = i + step
#     step_end = step_end if step_end < len(test) else len(test)
#     _test = fp.fit_transform(test.iloc[i:step_end])
#     test_t = np.vstack((test_t, _test))
#     print("test_t shape: ", test_t.shape)
# test = test_t



#np.save("./DataSet/testEngineered", test)

test = np.load("./DataSet/testEngineered.npy")
testCSV = pd.DataFrame(test, index=None, columns=None)
#print(testCSV)
testCSV.to_csv("./DataSet/testEngineered.csv",index=False)
#print(testCSV)


file_pre = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')

denom = 0
fold = 1 #Change to 5, 1 for Kaggle Limits
for i in range(fold):
    params = {
#         'eta': 0.03333,
        'eta': 0.02,
#         'max_depth': 4,
        'max_depth': 6,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'seed': i,
        'silent': True
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.20, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
    score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
    print(score1)
    #if score < 0.9:
    if denom != 0:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
        preds += pred
    else:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
        preds = pred.copy()
    denom += 1
#     submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
#     submission['ID'] = pid
#     submission.to_csv('./result/submission_xgb_fold_'  + str(i) + '_' + file_pre + '.csv', index=False)
preds /= denom
submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
print("Submissions: ",submission)
submission.to_csv('./result/submission_xgb_' + file_pre + '.csv', index=False)

import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['figure.figsize'] = (7.0, 7.0)
xgb.plot_importance(booster=model,); plt.show()




# train1 = pd.DataFrame(train[:170])
# train2 = pd.DataFrame(train[170:340])
# train3 = pd.DataFrame(train[340:510])
# train4 = pd.DataFrame(train[510:680])
# train5 = pd.DataFrame(train[680:850])
# train6 = pd.DataFrame(train[850:1020])
# train7 = pd.DataFrame(train[1020:1190])
# train8 = pd.DataFrame(train[1190:1360])
# train9 = pd.DataFrame(train[1360:1530])
# train10 = pd.DataFrame(train[1530:1700])
# train11 = pd.DataFrame(train[1700:1870])
# train12 = pd.DataFrame(train[1870:2040])
# train13 = pd.DataFrame(train[2040:2210])
# train14 = pd.DataFrame(train[2210:2380])
# train15 = pd.DataFrame(train[2380:2550])
# train16 = pd.DataFrame(train[2550:2720])
# train17 = pd.DataFrame(train[2720:2890])
# train18 = pd.DataFrame(train[2890:3060])
# train19 = pd.DataFrame(train[3060:3230])
# train20 = pd.DataFrame(train[3230:])
#
# trainList = [train1,train2,train3,train4,train5,train6,train7,train8,train9,train10,
#              train11,train12,train13,train14,train15,train16,train17,train18,train19,train20]
#
# pool = Pool()
# train = pool.map(fp.fit_transform, trainList)
# pool.close()
# pool.join()


# *java -Xmx4048m -jar StackNet.jar train task=classification has_head=true train_file=trainEngineered.csv test_file=testEngineered.csv params=parameters.txt sparse=False pred_file=querry_pred.csv test_target=false verbose=true Threads=10 stackdata=false folds=5 seed=1 metric=logloss*