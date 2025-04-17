# Hyper-parameter tuning without using MLFlow
import mlflow
import dagshub
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# initialize dagshub repository
dagshub.init(repo_owner='suryansh-sinha', repo_name='MLFlow-Basics')

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

# Using dagshub for experiment tracking
mlflow.set_tracking_uri('https://dagshub.com/suryansh-sinha/MLFlow-Basics.mlflow')
mlflow.set_experiment('breast-cancer-rf-hp')

with mlflow.start_run():
    grid_search.fit(X_train, y_train)
    
    # Log all the child runs
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric('Accuracy', grid_search.cv_results_['mean_test_score'][i])
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Log params
    mlflow.log_params(best_params)
    
    # Log metrics
    mlflow.log_metric("Accuracy", best_score)
    
    # Log training data
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, 'training data')
    
    # Log test data
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, 'testing data')
    
    # Log source code
    mlflow.log_artifact(__file__)
    
    # Log the best model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "Best Random Forest Model")
    
    # Set tags
    mlflow.set_tag('Author', 'Suryansh Sinha')
    
    print(best_params)
    print(best_score)