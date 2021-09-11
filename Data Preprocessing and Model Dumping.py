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
