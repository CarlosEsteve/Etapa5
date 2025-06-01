
import pandas as pd
import numpy as np
import json

import argparse
from pathlib import Path

import mlflow

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential


def main(args):
    mlflow.autolog()

    metrics = read_metrics(args.metrics_input)
    register_model(args.model_input, metrics, args.model_name)


def read_metrics(data_path):
    # Leer las métricas desde el archivo
    with open(data_path, "r") as f:
        metrics = json.load(f)

    print(f"Métricas leídas:", metrics)

    return metrics


def register_model(model_path, metrics, model_name):
    # Inicializar el cliente de Azure ML
   
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())

    accuracy = metrics.get('accuracy', 0)
    f1_score_weighted = metrics.get('f1_score_weighted', 0)
    threshold_accuracy = 0.80
    threshold_f1 = 0.75

    # Registrar el modelo si cumple los umbrales
    if accuracy >= threshold_accuracy and f1_score_weighted >= threshold_f1:
        registered_model = ml_client.models.create_or_update(
            Model(
                path=model_path,
                name=model_name,
                description=f"Modelo registrado con accuracy={accuracy}, f1_score_weighted={f1_score_weighted}"
            )
        )
        print(f"Modelo {model_name} registrado con éxito.")
    else:
        print(f"El modelo no cumple los umbrales: accuracy ({accuracy}) >= {threshold_accuracy} y f1_score_weighted ({f1_score_weighted}) >= {threshold_f1}. No se registrará.")
        raise ValueError(f"El modelo no cumple los umbrales: accuracy ({accuracy}) >= {threshold_accuracy} y f1_score_weighted ({f1_score_weighted}) >= {threshold_f1}. No se registrará.")




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", dest='model_input', type=str, required=True)
    parser.add_argument("--metrics_input", dest='metrics_input', type=str, required=True)
    parser.add_argument("--model_name", dest='model_name', type=str, required=True)
    args = parser.parse_args()

    return args


# run script
if __name__ == "__main__":
    args = parse_args()
    print('args:', args)
    main(args)
