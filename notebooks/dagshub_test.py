import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/Pakistan971/mlops_water.mlflow")

dagshub.init(repo_owner='Pakistan971', repo_name='mlops_water', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)