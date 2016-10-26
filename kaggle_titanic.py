import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
from sklearn import preprocessing, ensemble , cross_validation

sns.set_style('whitegrid')


def get_minor_and_girl(df_age, df_sex) :
	
	is_minor = []
	is_girl_minor = []

	for idx, _age in enumerate(df_age):
		if _age <= 18:
			is_minor.append(1)
		else:
			is_minor.append(0)

		if _age <= 18 and df_sex.iloc[idx] == 0:
			is_girl_minor.append(1)
		else:
			is_girl_minor.append(0)

	return is_minor, is_girl_minor


df_train = pd.read_csv('titanic_train.csv')
df_test = pd.read_csv('titanic_test.csv')
df_test_sub = df_test.drop(['Name','Ticket'], axis=1)

# drop unnecessary data(features) from the dataset
df_train = df_train.drop(['PassengerId','Name','Ticket'], axis=1)
df_test  = df_test.drop(['PassengerId','Name','Ticket'], axis=1)

# fill missing values in embarked column in training dataframe with most occuring value
df_train['Embarked'] = df_train['Embarked'].fillna('S')

#plotting various columns --> sex, Pclass seem to have greater control over survival rate.
# sns.factorplot('Parch','Survived', data=df_train)
# sns.plt.show()

#cleansing data in dataframe columns.
df_train['Cabin'].fillna('U', inplace=True)
df_test['Cabin'].fillna('U', inplace=True)

df_train['Cabin'] = df_train['Cabin'].map(lambda c : c[0])
df_test['Cabin'] = df_test['Cabin'].map(lambda c : c[0])

df_train['Age'].fillna(df_train['Age'].median(), inplace=True)
df_test['Age'].fillna(df_test['Age'].median(), inplace=True)

#Handling categorial columns through label encoder.
CATEGORIAL_COLUMNS = ['Sex', 'Embarked', 'Cabin']

le = preprocessing.LabelEncoder()

for col in CATEGORIAL_COLUMNS:
	data = df_train[col].append(df_test[col])
	le.fit(data.values)
	df_train[col] = le.transform(df_train[col])
	df_test[col] = le.transform(df_test[col])

df_train.dropna(inplace=True)
df_test.fillna( -99999, inplace=True)

#Feature Engineering TODO: bring in more features.
is_minor_train, is_girl_minor_train = get_minor_and_girl(df_train['Age'], df_train['Sex'])
is_minor_test, is_girl_minor_test = get_minor_and_girl(df_test['Age'], df_test['Sex'])

df_train['Is Minor'] = is_minor_train
df_train['Is Girl Minor'] = is_girl_minor_train
df_test['Is Minor'] = is_minor_test
df_test['Is Girl Minor'] = is_girl_minor_test

X = np.array(df_train.drop('Survived', 1))
y = np.array(df_train['Survived'])

#cross_validation for testing accuracy.
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#Random forest classifier
clf = ensemble.RandomForestClassifier()
clf.fit(X, y)

test_data = np.array(df_test)

#predict and export the data as csv.
output = clf.predict(test_data).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = df_test_sub['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)
