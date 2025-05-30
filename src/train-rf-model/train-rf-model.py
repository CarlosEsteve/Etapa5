
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import argparse
from pathlib import Path

import mlflow
import glob


def main(args):
    mlflow.autolog()

    df = get_data(args.input_data)
    model, X_test, y_test = train_model('Tipo_Mantenimiento', df, args.n_estimators, 0.2)

    mlflow.sklearn.save_model(model, args.model_output)

    X_test.to_csv((Path(args.output_data) / "mantenimiento_X_test.csv"), index = False)
    y_test.to_csv((Path(args.output_data) / "mantenimiento_y_test.csv"), index = False)

    

def get_data(data_path):

    all_files = glob.glob(data_path + "/*.csv")
    df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)
    
    return df


def train_model(target, data, n_estimators, test_size=0.2):
    # 4. Separar características (X) y objetivo (y)
    X = data.drop(columns=[target])  # Todas las columnas excepto target
    y = data[target]  # Variable objetivo

    # 5. Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    # 6. Preprocesamiento de datos
    # Identificar columnas numéricas y categóricas
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Crear un preprocesador con StandardScaler para numéricas y OneHotEncoder para categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # 7. Crear un pipeline con el preprocesador y el modelo rf
    mlflow.log_param("n_estimators:", n_estimators)

    rf_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # n_estimators es el parámetro de regularización
        ('regressor', RandomForestClassifier(n_estimators=n_estimators, random_state=42))
    ])

    # 8. Entrenar el modelo
    rf_model.fit(X_train, y_train)

    return rf_model, X_test, y_test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", dest='input_data', type=str)
    parser.add_argument("--n_estimators", dest='n_estimators', type=int, default=100)
    
    parser.add_argument("--model_output", dest='model_output',type=str)
    parser.add_argument("--output_data", dest='output_data',type=str)
    args = parser.parse_args()

    return args


# run script
if __name__ == "__main__":
    args = parse_args()
    print('args:', args)
    main(args)
