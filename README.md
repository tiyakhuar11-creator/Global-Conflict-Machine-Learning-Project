# Global Conflict Prediction Using Machine Learning

## Project Overview

This project uses Machine Learning to predict UN involvement in global conflicts using Logistic Regression. It involves data preprocessing, feature selection, encoding, scaling, model training, and evaluation using Python and Scikit-learn.

## Dataset Information

The original dataset contains approximately 3,000 records and 30 columns.

- Original columns: 30
- Selected columns: 24
- Target variable: `UN_Involvement`
- Problem type: Classification

## Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## Project Workflow

### 1. Data Loading
Loaded the global conflict dataset using Pandas and checked its structure and columns.

### 2. Handling Missing Values
Checked the dataset for missing values and identified 3 columns containing null values. Filled the missing values in all three columns using Mode to maintain the most frequently occurring values in the dataset.

### 3. Feature Selection
Created a new DataFrame named `df1` and selected 24 useful columns from the original 30 columns.

### 4. Label Encoding
Used Label Encoding to convert categorical text values into numerical values for machine learning.

### 5. Splitting Features and Target
Separated independent and dependent variables.

- **X:** Independent variables (selected features).
- **y:** Target variable (`UN_Involvement`).

```python
X = df1.drop('UN_Involvement', axis=1)
y = df1['UN_Involvement']
```

### 6. Train-Test Split
Used `train_test_split` from Scikit-learn to divide the dataset into training and testing sets:

- X_train
- X_test
- y_train
- y_test

### 7. Feature Scaling
Applied `StandardScaler` to scale the input features before training the model.

### 8. Model Building
Used Logistic Regression from `sklearn.linear_model` to build the classification model.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

### 9. Prediction
Used the trained model to predict UN involvement.

```python
y_pred = model.predict(X_test)
```

### 10. Model Evaluation
Evaluated model performance using:

- Confusion Matrix
- Classification Report
- Accuracy Score

## Libraries Used

| Library | Purpose |
|---|---|
| Pandas | Data handling |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| StandardScaler | Feature scaling |
| LogisticRegression | Classification |
| Jupyter Notebook | Code development |

## Key Learnings

- Handling missing numerical values.
- Creating a new DataFrame and selecting relevant columns.
- Applying Label Encoding.
- Feature selection and data splitting.
- Feature scaling using StandardScaler.
- Building a Logistic Regression model.
- Evaluating model performance.

## Conclusion

This project helped me gain practical experience in building a Machine Learning classification model. I learned how to preprocess real-world data, select relevant features, apply Logistic Regression, and evaluate model predictions using Scikit-learn.

## Author

**Tiya Khuar**
