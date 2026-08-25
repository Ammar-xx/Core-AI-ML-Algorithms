from sklearn.model_selection import train_test_split    #this imports the split function needed to prepare train and test data
from sklearn.linear_model import LinearRegression     #this imports the ML model, using linear as the target col is numerical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler       
import numpy as np
from sklearn.metrics import (       #metrics for numerical target data
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.datasets import fetch_california_housing


housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(df.head())

for col in df:
    df[col].fillna(df[col].mean(),inplace=True)

X=df.drop(columns=['MedHouseVal'])
Y=df['MedHouseVal']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=LinearRegression()
model.fit(X_train,Y_train)
pred=model.predict(X_test)

print(f"MAE: {mean_absolute_error(Y_test, pred):.3f}")

print(f"MSE: {mean_squared_error(Y_test, pred):.3f}")

print(f"RMSE: {np.sqrt(mean_squared_error(Y_test, pred)):.3f}")

print(f"R² Score: {r2_score(Y_test, pred):.3f}")