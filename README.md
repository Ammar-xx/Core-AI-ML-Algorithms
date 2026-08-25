# Core Machine Learning Algorithms

A collection of fundamental **Machine Learning algorithms implemented in Python using Scikit-learn and XGBoost**. This repository focuses on understanding the core workflow of supervised learning, including data preprocessing, feature scaling, model training, evaluation, and hyperparameter tuning.

## 📌 Algorithms Covered

### Regression

* **Linear Regression**
* **Decision Tree Regression**
* **Random Forest Regression**

### Classification

* **Logistic Regression**
* **XGBoost Classification**

## 📂 Project Structure

```text
Core_ML_Algos/
│
├── Linear_Regression.py
├── Logistic_Regression.py
├── DT.py
├── Random_Forest.py
├── XGB.py
│
├── EDA.png
├── EDA_sol.png
├── ML_Models.png
├── Model_Classification.png
├── Scalar_Table.png
└── Vis_Task.png
```

## 📊 Datasets

The implementations use built-in and easily accessible datasets:

* **California Housing Dataset** — used for Linear Regression
* **Iris Dataset** — used for Logistic Regression and XGBoost Classification
* **Diabetes Dataset** — used for Decision Tree and Random Forest Regression

## 🔧 Concepts Demonstrated

### Data Preprocessing

* Train-test splitting
* Handling missing values
* Feature scaling with `StandardScaler`
* Label encoding for categorical target variables
* Outlier detection and clipping
* Basic Exploratory Data Analysis (EDA)

### Model Training

Each implementation demonstrates the basic workflow:

```text
Load Dataset
     ↓
EDA & Data Preprocessing
     ↓
Separate Features and Target
     ↓
Train-Test Split
     ↓
Feature Scaling / Encoding
     ↓
Train Model
     ↓
Make Predictions
     ↓
Evaluate Model
```

### Hyperparameter Tuning

The project also demonstrates **GridSearchCV** for finding suitable model hyperparameters.

For example, Decision Tree and Random Forest models are tuned using parameters such as:

* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `n_estimators`

Cross-validation is performed using **5-fold cross-validation**, with **R² score** used for regression model selection.

## 📈 Model Evaluation

### Regression Metrics

The regression models are evaluated using:

* **MAE (Mean Absolute Error)**
* **MSE (Mean Squared Error)**
* **RMSE (Root Mean Squared Error)**
* **R² Score**

### Classification Metrics

The classification models use:

* **Accuracy**
* **Precision**
* **Recall**
* **F1 Score**
* **Confusion Matrix**
* **Classification Report**

## 🧠 Key Learning Points

This repository is intended to provide practical understanding of how different ML algorithms work and how they are applied to real datasets.

Some of the concepts explored include:

* Difference between **regression and classification**
* Choosing appropriate evaluation metrics
* Importance of feature scaling
* Encoding categorical target variables
* Detecting and handling outliers
* Understanding model performance using EDA
* Avoiding data leakage during preprocessing
* Hyperparameter tuning with `GridSearchCV`
* Cross-validation
* Comparing training and testing performance

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost

## 🚀 Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Core_ML_Algos
```

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## ▶️ Running the Programs

Each algorithm is implemented as an individual Python file.

For example:

```bash
python Linear_Regression.py
```

or:

```bash
python Logistic_Regression.py
```

Similarly, you can run:

```bash
python DT.py
python Random_Forest.py
python XGB.py
```

## 📚 Purpose

This project was created as a hands-on collection for learning and practicing **core Machine Learning algorithms and workflows**. It serves as a reference for understanding how preprocessing, model training, evaluation, and hyperparameter optimization come together in a typical ML pipeline.

## 🔮 Future Improvements

* Add more classification and regression algorithms
* Add visual model comparisons
* Improve reusable preprocessing pipelines
* Add automated EDA
* Add ROC-AUC and Precision-Recall curves
* Compare models using a common evaluation framework
* Add notebooks with detailed explanations
* Experiment with larger real-world datasets

## 👨‍💻 Author

**Muhammad Ammar**

A learning-focused Machine Learning repository covering fundamental algorithms and practical ML workflows.
