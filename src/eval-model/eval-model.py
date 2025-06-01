
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
import argparse
from pathlib import Path
import mlflow

def main(args):
    mlflow.autolog()
    model = mlflow.sklearn.load_model(args.model_input)
    X_test, y_test = get_data(args.input_data)
    evaluate_model(model, X_test, y_test, args.metrics_output)

def get_data(data_path):
    X_test = pd.read_csv((Path(data_path) / "mantenimiento_X_test.csv"))
    y_test = pd.read_csv((Path(data_path) / "mantenimiento_y_test.csv"))
    return X_test, y_test

def evaluate_model(model, X_test, y_test, metrics_output):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score_weighted", f1)
        mlflow.sklearn.log_model(model, "model")

    # Guardar las métricas principales en un archivo JSON
    metrics = {
        "accuracy": acc,
        "f1_score_weighted": f1
    }
    with open(metrics_output, "w") as f:
        json.dump(metrics, f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", dest='model_input', type=str, required=True)
    parser.add_argument("--input_data", dest='input_data', type=str, required=True)
    parser.add_argument("--metrics_output", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print('args:', args)
    main(args)
