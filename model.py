# This file is showing how model process data form raw to training,
# for  pre-modelling there is saperate ipynb file
# importing Required libraries
import logging
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# log file initialization 
logging.basicConfig(filename='debug.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

logging.debug('Model.py File execution started ')

# loading database with pandas library
df = pd.read_csv("./dataset/final.csv")
logging.debug(' Database Loaded ')

col_names = df.columns
category_col = ['name','location','type','city','cuisines']

labelEncoder = preprocessing.LabelEncoder()

mapping_dict = {}
for col in category_col:
	df[col] = labelEncoder.fit_transform(df[col])

	le_name_mapping = dict(zip(labelEncoder.classes_,
							labelEncoder.transform(labelEncoder.classes_)))

	mapping_dict[col] = le_name_mapping


# model featuring
X = df[['name',
 'onlineorder',
 'booktable',
 'location',
 'cuisines',
 'cost',
 'type',
 'city',
 'Mean Rating']]
y = df['rate']

# Data Spliting For model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# model fitting using LGBMClassifier
clf = lgb.LGBMRegressor(first_metric_only = True)
clf.fit(X_train, y_train)

# Printing Accuracy
y_pred = clf.predict(X_test)

from sklearn.metrics import mean_squared_error
logging.debug('Accuracy: {}'.format(mean_squared_error(y_test,y_pred)))


# pkl export & finish log
pickle.dump(clf, open("model.pkl", "wb"))
logging.debug(' Execution of Model.py is finished ')