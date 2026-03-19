import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv(r"C:\Users\Admin\Desktop\loan_data.csv")

print("First 5 Rows: hyat data add karu shakto\n", df.head())
print("\nDataset Info:\n")
print(df.info())

print("\nMissing Values:\n", df.isnull().sum())

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# datatype nusar column select karti tyasathi select_dtype(include=['']).columnsha syntax ahe

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Numerical Pipeline
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Categorical Pipeline
cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Column Transformer
preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)

y_pred_log = model.predict(X_test)

print("\n=== Logistic Regression Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))


# 8. Random Forest Model
rf_model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    # accuracy bhette multiple trees var work karta yeta te aplyala useful hota'
    ("classifier", RandomForestClassifier(random_state=42))
])

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n=== Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
# 9. Cross Validation (Random Forest)
cv_scores = cross_val_score(rf_model, X, y, cv=5)

print("\nCross Validation Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())

# 10. Hyperparameter Tuning
param_grid = {
    "classifier__n_estimators": [50, 100],
    "classifier__max_depth": [None, 5, 10]
}

grid_search = GridSearchCV(
    rf_model,
    param_grid,
    cv=3,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)
# saglyat jasta accurate fayda denare paragms mhajech te ahe  fayda .best_params_ 
print("\nBest Parameters:", grid_search.best_params_)
#  best_estimator only best parametre sangto
best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)

print("\n=== Tuned Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("Classification Report:\n", classification_report(y_test, y_pred_best))

# new_data = pd.DataFrame({
#     "Gender": ["Male"],
#     "Married": ["Yes"],
#     "Dependents": ["0"],
#     "Education": ["Graduate"],
#     "Self_Employed": ["No"],
#     "ApplicantIncome": [5000],
#     "CoapplicantIncome": [2000],
#     "LoanAmount": [150],
#     "Loan_Amount_Term": [360],
#     "Credit_History": [1],
#     "Property_Area": ["Urban"]
# })

# prediction = model.predict(new_data)

# print("Loan Prediction:", prediction)