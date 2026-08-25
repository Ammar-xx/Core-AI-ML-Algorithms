from sklearn.model_selection import train_test_split    #this imports the split function needed to prepare train and test data
from sklearn.tree import DecisionTreeRegressor     #this imports the ML model, using linear as the target col is numerical  
import numpy as np
from sklearn.metrics import (       #metrics for numerical target data
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.datasets import load_diabetes
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

"""
print(df.head())
print(df.info())
print(f"Description:\n{df.describe()}")
print(f"Null values:\n{df.isnull().sum()}")
print(f"Duplicated:\n{df.duplicated().sum()}")


for col in df.columns:
    sns.boxplot(y=df[col])
    plt.show()

for col in df.columns:
    sns.histplot(data=df[col],kde=True)     #to check the skewness
    plt.show()

corr = df.corr(numeric_only=True)
plt.figure(figsize=(10,8))
sns.heatmap(                #to check the relationship between the target and the features
    corr,                   #acc to the heatmap, target has moderate positive relationship with: bmi,bp,s4,s5
    annot=True,             #and a little less than moderate negative relationship with s3
    cmap="coolwarm",
    fmt=".2f"
)
plt.show()
"""
X=df.drop(columns=['target'])
Y=df['target']

X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,train_size=0.8,random_state=42
)

# Columns with outliers
cols = ['s1', 's2', 's3', 's4', 's5', 's6']

# Compute clipping limits from TRAINING DATA ONLY
for col in cols:

    lower = X_train[col].quantile(0.05)
    upper = X_train[col].quantile(0.95)

    # Clip training data
    X_train[col] = X_train[col].clip(lower, upper)

    # Apply same limits to test data
    X_test[col] = X_test[col].clip(lower, upper)

model=DecisionTreeRegressor(random_state=42)

param_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    model, param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1           
)

grid_search.fit(X_train,Y_train)
pred=grid_search.predict(X_test)

best_model = grid_search.best_estimator_
print("Training R²:", best_model.score(X_train, Y_train))
print("Testing R² :", best_model.score(X_test, Y_test))
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)
print(f"MAE: {mean_absolute_error(Y_test, pred):.3f}")

print(f"MSE: {mean_squared_error(Y_test, pred):.3f}")

print(f"RMSE: {np.sqrt(mean_squared_error(Y_test, pred)):.3f}")

print(f"R² Score: {r2_score(Y_test, pred):.3f}")