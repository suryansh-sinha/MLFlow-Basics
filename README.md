# MLFlow-Basics
Experiment Tracking using MLFlow

MLFlow is used for experiment tracking. It's a better tool for tracking experiments than DVC, which recently added this feature. To use MLFlow, we install it using pip or brew.
`pip install mlflow`

To initialize our MLFlow project, we can type `mlflow ui --port 5000` in the terminal to host it locally on port 5000.

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load wine dataset
wine = load_wine()
X = wine.data
y = wine.target  

# Set the tracking URI
mlflow.set_tracking_uri('http://127.0.0.1:5000')

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  

# Define the params for RF model
max_depth = 5
n_estimators = 10

with mlflow.start_run():
	rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
	rf.fit(X_train, y_train)
	y_pred = rf.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	mlflow.log_metric('accuracy', accuracy)
	mlflow.log_param('max_depth', max_depth)
	mlflow.log_param('n_estimators', n_estimators)
	print(accuracy)
```

First, we have to set up our URI since by default, the URI is set to the location of the local directory on our pc and we want to change it so that we can access our artifacts on the locally hosted server. To do this we have the following block of code -
```python
# Set the tracking URI
mlflow.set_tracking_uri('http://127.0.0.1:5000')
```


## Logging Stuff
To track our experiment, we create a context manager using `{python}mlflow.start_run()`, then we can track the parameters, metrics, and artifacts inside it using -
- `mlflow.log_metric('Metric_Name', metric_variable_name)`
- `mlflow.log_param('Parameter_Name', parameter_variable_name)`
- `mlflow.log_artifact('Artifact_Name', artifact_directory)`
- `mlflow.sklearn.log_model(model, 'Model_Name')`

They have also given a convenient API to perform logging automatically. This can be done using - `mlflow.autolog()` - This automatically identifies all the parameters and metrics that need to be tracked. This also tracks the datasets, model, and image artifacts. This does not track the CODE FILE as an artifact. 

To track the code file you're working on, we can do -
- `mlflow.log_artifact(__file__)` - This accesses the current file and logs that as an artifact.

What `autolog` can track -
- Hyperparameters
- Metrics
- Model
- Artifacts - Model Summary and Plots
- Training Parameters
- Environment Information - requirements.txt
- Training Data and Labels - Not the entire dataset itself, but information about dataset size, feature information, etc
- Automatic Model Signature - Infers the type of model (signature) and saves that along with the model.
What it doesn't track -
- Custom Metrics - Metrics that we've created ourselves.
- Custom Artifacts - Plots, charts that are not a part of the model training process
- Preprocessed Data - Not logged unless done manually as an artifact.
- Intermediate Model States - Model checkpoints are not saved unless done specifically.
- Complex Model Structures - If we use a highly customized model structure, autolog might miss some logging details
- Custom Training Loops
- Non-supported frameworks
- Custom Hyperparameter Tuning
## To create a new experiment
Create a new experiment in the UI using the + button (optional). Then you can set the experiment name in the code before `with mlflow.start_run():` using - 
```python
# Experiment name should be the same as the one created in UI.
# If the experiment doesn't exist in UI, it will create it.
mlflow.set_experiment('Test-Exp1')
```

Another way to do this is to use the unique ID of the experiment directly as -
```python
with mlflow.start_run(experiment_id=650762004362524175):
	# Experiment Code Here.
```

## MLFlow Server Setup
So far, we've worked on MLFlow locally. All the files were being hosted to our local machine. We can also integrate a database for storing files such as the metadata and we can store our artifacts locally. This makes data storage really convenient for the user.

In a collaborative setting, MLFlow has something called the MLFlow Tracking Server, which needs to be hosted on something like an EC2 instance from AWS. This server can be accessed by the entire team, and everyone can collaborate and create experiments for the same project. The tracking server can then save our data to a remote cloud storage like AWS S3 or a local SQL (MySQL, PostgreSQL, SQLite) database, so it can be accessed by everyone who has access to the remote tracking server.

Setting up IAM, EC2, S3 and then configuring it to work as ML Flow Tracking Server and storage is a complex procedure. Most companies instead use something called as AWS dagshub. It's a platform where we don't have to explicitly setup the server, cloud storage etc. We just have to connect our repository and it does all these things for us automatically.

## Setting up Dagshub

Open Dagshub dashboard and create a new repository. We can import already existing repositories from Git. After creating a repository in Dagshub, we can perform experiment tracking using mlflow and we use dagshub to handle the tracking server, connecting to cloud storage etc.

To set up dagshub in a code file -
```python
import dagshub

# Initializing dagshub
dagshub.init(repo_owner='suryansh-sinha', repo_name='MLFlow-Basics', mlflow=True)

# Now, the tracking URL will be Dagshub's remote tracking URL
mlflow.set_tracking_uri('https://dagshub.com/suryansh-sinha/MLFlow-Basics.mlflow')
```
This code sets up dagshub as our remote tracking server where we can see all of our experiments. So, instead of seeing our experiments locally, we can now use the URL - https://dagshub.com/suryansh-sinha/MLFlow-Basics.mlflow to look at our experiments. To make changes collaboratively, we can just share an API key for the project and that allows us to perform multiple experiments on the same repository.
## Hyperparameter Tuning Using MLFlow -

Check out the code implementation in hypertune2.py in the src folder.
To log all the runs that are inside the grid search and compare results for different parameters, we track the experiments using `with mlflow.start_runs(nested=True) as child:` 

To track the inputs i.e. the dataset, we first create a copy of the dataset using pandas. Then we can use `mlflow.data.from_pandas(train_df)` to convert the dataset into an mlflow trackable artifact. Then we can just use `mlflow.log_input(train_df, 'training data')` to track the one of the inputs of our model.

## Model Registry

There are three stages to model development - 
- Development Stage (performing experiments to find out what could be the best model)
- Staging - Here, the model is being tested locally and by other teams.
- Production - The model has passed staging and is now ready for deployment as a product
- Retirement - Old models that were trained on older data or models with older architectures are retired (archived).

Model registry is basically just registering a model into a log kind of entry in MLFlow and then we can use different kinds of flags for the model. So, for the sake of an example, if we create a model registry for our model above, we can then choose the stage in which the model is i.e. Development, Staging, Production, Retirement etc. This helps us keep track of the model in a sensible way. Also helps move model to archive very conveniently.
