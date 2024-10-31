from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
import numpy as np
import pandas as pd


mlflow.set_tracking_uri("https://dagshub.com/Pakistan971/mlops_water.mlflow")
dagshub.init(repo_owner='Pakistan971', repo_name='mlops_water', mlflow=True)
mlflow.set_experiment("Experiment1")

data = pd.read_csv(r"C:\Users\Shahbaz\mlops_new\water_potability.csv")

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data,test_size=0.2,random_state=69)

def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value,inplace=True)
    return df

train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

from sklearn.ensemble import RandomForestClassifier
import pickle

X_train = train_processed_data.drop(columns=["Potability"],axis=1)
y_train = train_processed_data["Potability"]

n_estimators = 100


with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train,y_train)

    pickle.dump(clf,open("model.pkl","wb"))

    X_test = test_processed_data.iloc[:,0:-1].values
    y_test = test_processed_data.iloc[:,-1].values

    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

    model = pickle.load(open("model.pkl","rb"))

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)


    mlflow.log_metric("acc",acc)
    mlflow.log_metric("aprecisioncc",precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("f1_sccore",f1)

    mlflow.log_param("n_estimators",n_estimators)

    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")

    mlflow.sklearn.log_model(clf,"RandomForestClassifier")

    mlflow.log_artifact(__file__)

    mlflow.set_tag("author","Shahbaz")
    mlflow.set_tags({"model":"GB"})


    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1_score: {f1}")