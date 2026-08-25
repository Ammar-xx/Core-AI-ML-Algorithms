from sklearn.model_selection import train_test_split    #this imports the split function needed to prepare train and test data
from sklearn.linear_model import LogisticRegression     #this imports the ML model, using logistic as the target col is categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler       
import numpy as np
import seaborn as sns
from sklearn.metrics import (       #metrices for categorical target data
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

df=sns.load_dataset('iris')   #dataset we would be training on

"""
Standard scaler converts the mean of the numerical features to 0 and all the values below the mean are negative and above the mean are positive as many machine learning algorithms learn better when data is centered around zero.

MinMax Scaler converts the range of all the features into [0,1], as some algorithms perform best when all inputs are within a known range.

A picture is attached in the folder that shows that which model uses which scaler

Label Encoder is used to transform the target dataset (Y) from categorical into numerical value

for non target data, we use pd.dummies or map to convert it from categorical to numerical
"""

X=df.drop(columns=['species'])
Y=df['species']

sc=StandardScaler()     #description given above

le=LabelEncoder()
Y=le.fit_transform(Y)       #this converts the categorical into numerical so that the model can read it. We use it only for target data

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)

X_train=sc.fit_transform(X_train)   #use fit_transform for training data
X_test=sc.transform(X_test)         #use transform for test data

model=LogisticRegression()

model.fit(X_train,Y_train)
pred=model.predict(X_test)

print(f"Accuracy: {accuracy_score(Y_test, pred):.2f}")
print(f"Precision: {precision_score(Y_test, pred, average='weighted'):.2f}")        #we used weighted here as the target data is not binary
print(f"Recall: {recall_score(Y_test, pred, average='weighted'):.2f}")
print(f"F1 Score: {f1_score(Y_test, pred, average='weighted'):.2f}")

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, pred))

print("\nClassification Report:")
print(classification_report(Y_test, pred))