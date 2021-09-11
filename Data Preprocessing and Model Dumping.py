#importing libreries
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import add
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score,precision_score,classification_report,roc_auc_score,roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import joblib
warnings.filterwarnings('ignore')

#data_loading
data = pd.read_csv('Dataset\CHD_Dataset.csv')
data.drop(['education'], axis=1, inplace=True)
data.dropna(axis=0, inplace=True)
data.shape

#define the features
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

forest = RandomForestClassifier(n_estimators=1000, n_jobs= 1, class_weight='balanced')

#define Boruta feature selection
feat_selector = BorutaPy(forest, n_estimators= 'auto', verbose=2)

#find all relevant features
feat_selector.fit(x,y)

most_important = data.columns[:-1][feat_selector.support_].tolist()

top_features = data.columns[:-1][feat_selector.ranking_ <=6].tolist()

X_top = data[top_features]
y = data['TenYearCHD']

res = sm.Logit(y,X_top).fit()

params = res.params
conf = res.conf_int()
conf['Odds Ratio'] = params
conf.columns = ['5%', '95%', 'Odds Ratio']

X = data[top_features]
y = data.iloc[:,-1]

# the numbers before smote
num_before = dict(Counter(y))

#perform smoting

# define pipeline
over = SMOTE(sampling_strategy=0.8)
under = RandomUnderSampler(sampling_strategy=0.8)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

# transform the dataset
X_smote, y_smote = pipeline.fit_resample(X, y)


#the numbers after smote
num_after =dict(Counter(y_smote))

# new dataset
new_data = pd.concat([pd.DataFrame(X_smote), pd.DataFrame(y_smote)], axis=1)
new_data.columns = ['age', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose','TenYearCHD']

X_new = new_data[top_features]
y_new= new_data.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X_new,y_new,test_size=.2,random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.transform(X_test)
X_test = pd.DataFrame(X_test_scaled)

#grid search for optimum parameters
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
svm_clf = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=10)

# train the model
svm_clf.fit(X_train,y_train)

svm_predict = svm_clf.predict(X_test)
#accuracy
svm_accuracy = accuracy_score(y_test,svm_predict)
print(f"Using SVM we get an accuracy of {round(svm_accuracy*100,2)}%")

filename = 'final_model.sav'
import joblib
joblib.dump(svm_clf, filename)