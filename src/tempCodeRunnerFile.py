# Creating our model
rf = RandomForestClassifier(random_state=42)

# Defining our parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
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