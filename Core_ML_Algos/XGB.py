from sklearn.model_selection import train_test_split    #this imports the split function needed to prepare train and test data
from xgboost import XGBClassifier     #this imports the ML model, using linear as the target col is numerical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler       
import numpy as np
from sklearn.metrics import (       #metrices for categorical target data
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

df=sns.load_dataset('iris')

X=df.drop(columns=['species'])
Y=df['species']

sc=StandardScaler()     #description given above

le=LabelEncoder()
Y=le.fit_transform(Y)       #this converts the categorical into numerical so that the model can read it. We use it only for target data

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)

X_train=sc.fit_transform(X_train)   #use fit_transform for training data
X_test=sc.transform(X_test)         #use transform for test data


model=XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    eval_metric='mlogloss',
    early_stopping_rounds=50
)

model.fit(X_train,Y_train,eval_set=[(X_test,Y_test)],verbose=True)

pred=model.predict(X_test)

print(f"Accuracy: {accuracy_score(Y_test, pred):.2f}")
