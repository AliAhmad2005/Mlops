from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


mlflow.set_tracking_uri("https://dagshub.com/Pakistan971/mlops_water.mlflow")
dagshub.init(repo_owner='Pakistan971', repo_name='mlops_water', mlflow=True)
mlflow.set_experiment("Experiment3")

data = pd.read_csv(r"C:\Users\Shahbaz\mlops_new\water_potability.csv")
train_data, test_data = train_test_split(data,test_size=0.2,random_state=69)

def fill_missing_with_mean(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].mean()
            df[column].fillna(median_value,inplace=True)
    return df

train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)


X_train = train_processed_data.drop(columns=["Potability"],axis=1)
y_train = train_processed_data["Potability"]
X_test = test_processed_data.drop(columns=["Potability"],axis=1)
y_test = test_processed_data["Potability"]

model = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest" : RandomForestClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbor":KNeighborsClassifier(),
    "XG Boost":XGBClassifier()

}

with mlflow.start_run(run_name="Water Potability Models Experiment"):
    #running for each model in model dict
    for model_name, model in model.items():
        with mlflow.start_run(run_name=model_name,nested=True):
            model.fit(X_train,y_train)
            model_filename = f"{model_name.replace(' ','_')}.pkl"

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)


            mlflow.log_metric("acc",acc)
            mlflow.log_metric("aprecisioncc",precision)
            mlflow.log_metric("recall",recall)
            mlflow.log_metric("f1_sccore",f1)

            cm = confusion_matrix(y_test,y_pred)
            plt.figure(figsize=(5,5))
            sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix fro {model_name}")

            plt.savefig(f"{model_name.replace(' ','_')}.png")
            mlflow.log_artifact(f"{model_name.replace(' ','_')}.png")
            mlflow.sklearn.log_model(model,model_name.replace(' ','_'))
            mlflow.log_artifact(__file__)
            mlflow.set_tag("author","Shahbaz")


    print("All Models have been trained and logged as Child Run Successfully.")