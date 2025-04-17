# Hyper-parameter tuning without using MLFlow
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating our model
rf = RandomForestClassifier(random_state=42)

# Defining our parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [16, 32, 64, 128],
    'max_depth': [None, 10, 32, 64]
}

# Applying GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Hyperparameter tuning on our parameters without using MLFlow.
grid_search.fit(X_train, y_train)

# Displaying the best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(best_params)
print(best_score)