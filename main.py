import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

# Load data
data = pd.read_csv('generated_medical_dataset.csv')

# Drop the 'Patient No' column
#data = data.drop(columns=['Patient No'])

# Define feature columns and target column
feature_columns = data.columns[:-1]
target_column = 'Disease'

# Correct target mapping if needed
unique_classes = sorted(data[target_column].unique())
class_mapping = {v: i for i, v in enumerate(unique_classes)}
data[target_column] = data[target_column].map(class_mapping)

# Split data into features and target
X = data[feature_columns]
y = data[target_column]

# Identify categorical and numeric columns
categorical_cols = ['Gender', 'Medications', 'Food']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline for numeric and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Ensure that train/test split includes all classes using stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Transform the data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Define models and hyperparameters
models = {
    'Logistic Regression': {
        'model': LogisticRegression(),
        'params': {'C': [0.01, 0.1, 1, 10, 100]}
    },
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30], 'min_samples_split': [2, 5, 10]}
    },
    'XGBoost': {
        'model': XGBClassifier(),
        'params': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}
    },
    'LightGBM': {
        'model': LGBMClassifier(),
        'params': {'n_estimators': [50, 100, 300], 'max_depth': [-1, 10, 20], 'learning_rate': [0.01, 0.1, 0.2]}
    }
}

best_estimators = {}
skf = StratifiedKFold(n_splits=3)

for model_name, model_info in models.items():
    clf = GridSearchCV(model_info['model'], model_info['params'], cv=skf, n_jobs=-1)
    clf.fit(X_train_transformed, y_train)
    best_estimators[model_name] = clf.best_estimator_
    print(f"{model_name} best parameters: {clf.best_params_}")
    print(f"{model_name} Accuracy: {clf.score(X_test_transformed, y_test)}")

# Ensemble model
ensemble_model = VotingClassifier(estimators=[
    ('lr', best_estimators['Logistic Regression']),
    ('rf', best_estimators['Random Forest']),
    ('xgb', best_estimators['XGBoost']),
    ('lgbm', best_estimators['LightGBM'])
], voting='soft')

ensemble_model.fit(X_train_transformed, y_train)
print(f"Ensemble Model Accuracy: {ensemble_model.score(X_test_transformed, y_test)}")

# Save models and preprocessor
joblib.dump(ensemble_model, 'best_ensemble_model.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')