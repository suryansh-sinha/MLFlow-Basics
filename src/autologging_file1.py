import mlflow
import dagshub
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Initializing dagshub
dagshub.init(repo_owner='suryansh-sinha', repo_name='MLFlow-Basics',)

# Set the tracking URI
# mlflow.set_tracking_uri('http://127.0.0.1:5000')  LOCAL
mlflow.set_tracking_uri('https://dagshub.com/suryansh-sinha/MLFlow-Basics.mlflow')

# Load wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define the params for RF model
max_depth = 16
n_estimators = 32

# Setting which experiment the run goes inside.
mlflow.autolog()
mlflow.set_experiment('Test-Exp1')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Creating and saving a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_mat.png')
    
    # Log artifacts (not covered by autolog)
    mlflow.log_artifact(__file__)
    
    # Tags
    mlflow.set_tags({'Author':'Suryansh', 'Project':'Wine Classification'})
        
    print(accuracy)