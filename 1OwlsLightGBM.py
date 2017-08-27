from sklearn import preprocessing, pipeline, feature_extraction, decomposition, model_selection, metrics, cross_validation, svm
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
import lightgbm as lgb
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier


import datetime

train = pd.read_csv("./DataSet/training_variants")
test = pd.read_csv("./DataSet/test_variants")
trainx = pd.read_csv("./DataSet/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
testx = pd.read_csv("./DataSet/test_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])

train = pd.merge(train, trainx, how='left', on='ID').fillna('')
y = train['Class'].values
train = train.drop(['Class'], axis=1)

test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values

df_all = pd.concat((train, test), axis=0, ignore_index=True)
df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)

gen_var_lst = sorted(list(train.Gene.unique()) + list(train.Variation.unique()))
print(len(gen_var_lst))

gen_var_lst = [x for x in gen_var_lst if len(x.split(' '))==1]
print(len(gen_var_lst))
i_ = 0

#commented for Kaggle Limits
for gen_var_lst_itm in gen_var_lst:
    if i_ % 100 == 0: print(i_)
    df_all['GV_'+str(gen_var_lst_itm)] = df_all['Text'].map(lambda x: str(x).count(str(gen_var_lst_itm)))
    i_ += 1

for c in df_all.columns:
    if df_all[c].dtype == 'object':
        if c in ['Gene','Variation']:
            lbl = preprocessing.LabelEncoder()
            df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
        elif c != 'Text':
            lbl = preprocessing.LabelEncoder()
            df_all[c] = lbl.fit_transform(df_all[c].values)
        if c=='Text':
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))

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
                                        decomposition.TruncatedSVD(n_components=20, n_iter=30, random_state=12))])),
            ('pi2', pipeline.Pipeline([('Variation', cust_txt_col('Variation')),
                                       ('count_Variation',
                                        feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))),
                                       ('tsvd2',
                                        decomposition.TruncatedSVD(n_components=20, n_iter=30, random_state=12))])),
            # commented for Kaggle Limits
            ('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')),
                                        ('hv', feature_extraction.text.HashingVectorizer(decode_error='ignore', n_features=2 ** 16, non_negative=True, ngram_range=(1, 4))),
                                        ('tfidf_Text', feature_extraction.text.TfidfTransformer()),
                                        ('tsvd3', decomposition.TruncatedSVD(n_components=500, n_iter=30, random_state=12))]))
        ])
     )])

train = fp.fit_transform(train)
print(train.shape)



np.save("./DataSet/trainEngineeredBigger", train)


test_t = np.empty([0, train.shape[1]])
step = 200
for i in range(0, len(test), step):
    step_end = i + step
    step_end = step_end if step_end < len(test) else len(test)
    _test = fp.transform(test.iloc[i:step_end])
    test_t = np.vstack((test_t, _test))
test = test_t
print(test.shape)

np.save("./DataSet/trainEngineeredBigger", test)

train = np.load("./DataSet/trainEngineeredBigger.npy")
test = np.load("./DataSet/trainEngineeredBigger.npy")
file_pre = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')

y = y - 1 #fix for zero bound array
denom = 0
fold = 1 #Change to 5, 1 for Kaggle Limits

param = {'boosting':'gbdt',
         'num_leaves':15,
         'num_trees':100,
         'objective':'multiclass',
         'learning_rate':0.02,
         'num_class': 9,
         'max_bin': 255,
         'min_sum_hessian_in_leaf':0.001,
         'bagging_fraction':1,
         'lambda_l1':0,
         'num_iterations':500
         }

param['metric'] = 'multi_logloss'

num_round = 3000
early_stopping_rounds = 20

x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.01, random_state=7)

train_data = lgb.Dataset(x1, label=y1)
dev_data = lgb.Dataset(x2, label=y2)

bst = lgb.train(param, train_data, num_round, valid_sets=[dev_data], early_stopping_rounds = early_stopping_rounds)

ypred = bst.predict(test)

submission = pd.DataFrame(ypred, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('./result/submission_lgb_' + file_pre + '.csv', index=False)

import matplotlib.pyplot as plt
import seaborn as sns

# plt.rcParams['figure.figsize'] = (7.0, 7.0)
# xgb.plot_importance(booster=model,); plt.show()