
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json

import argparse
from pathlib import Path

import mlflow


def main(args):
    mlflow.autolog()

    model = mlflow.sklearn.load_model(args.model_input)
    X_test, y_test = get_data(args.input_data)
    evaluate_model(model, X_test, y_test)


def get_data(data_path):
    X_test = pd.read_csv((Path(data_path) / "mantenimiento_X_test.csv"))
    y_test = pd.read_csv((Path(data_path) / "mantenimiento_y_test.csv"))

    return X_test, y_test


def evaluate_model(a_model, X_test, y_test):
    # 9. Hacer predicciones
    y_pred = a_model.predict(X_test)

    # 10. Evaluar el modelo usando MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"Error Cuadrático Medio (MSE): {mse:.2f}")

    # 11. Calcular la varianza de los datos objetivo
    variance = np.var(y_test.values)
    print(f"Varianza de los datos objetivo: {variance:.2f}")

    # 12. Comparación entre MSE y varianza
    mse_to_variance_ratio = mse / variance
    print(f"Relación MSE/Varianza: {mse_to_variance_ratio:.2f}")

    with mlflow.start_run():
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mse2var", mse_to_variance_ratio)
        mlflow.sklearn.log_model(a_model, "model")

    # Guardar las métricas en un archivo JSON
    save_metrics(mse, mse_to_variance_ratio, args.metrics_output)


def save_metrics(mse, mse_to_variance_ratio, output_path):
    metrics = {
        "mse": mse,
        "mse_to_variance_ratio": mse_to_variance_ratio
    }
    
    with open(output_path, "w") as f:
        json.dump(metrics, f)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", dest='model_input', type=str, required=True)
    parser.add_argument("--input_data", dest='input_data', type=str, required=True)
    parser.add_argument("--metrics_output", type=str, required=True)
    args = parser.parse_args()

    return args


# run script
if __name__ == "__main__":
    args = parse_args()
    print('args:', args)
    main(args)
