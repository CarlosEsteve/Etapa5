
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import mlflow
import glob
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def main(args):
    mlflow.autolog()
    df = get_data(args.input_data)
    X_test, y_test, y_pred, clf = entrenar_modelo(df, args.horizon, args.n_estimators)
    mlflow.sklearn.save_model(clf, args.model_output)
    # Guardar conjuntos de test y predicciones
    X_test.to_csv((Path(args.output_data) / "mantenimiento_X_test.csv"), index=False)
    y_test.to_csv((Path(args.output_data) / "mantenimiento_y_test.csv"), index=False)
    pd.DataFrame({'y_pred': y_pred}).to_csv((Path(args.output_data) / "mantenimiento_y_pred.csv"), index=False)
    print("Entrenamiento y guardado finalizado.")

def get_data(input_folder):
    # Lee todos los CSV de la carpeta y los concatena
    all_files = glob.glob(input_folder + "/*.csv")
    df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)
    return df

def entrenar_modelo(df, horizon, n_estimators):
    # 1. Variables principales
    data = df['Horas_Operativas']
    target = df['Tipo_Mantenimiento']

    # 2. Modelo de regresión para tendencia
    reg = RandomForestRegressor(n_estimators=100, random_state=123)
    reg.fit(np.arange(len(data)).reshape(-1, 1), data)
    tendencia = reg.predict(np.arange(len(data)).reshape(-1, 1))

    # 3. Residuales
    residuales = data.values - tendencia

    # 4. Ingeniería de características: lags y residuales
    max_lag = min(470, len(data) - horizon - 1)
    lags = [data.shift(lag).rename(f'lag_{lag}') for lag in range(1, max_lag + 1)]
    X = pd.concat(lags, axis=1)
    X['residuals'] = residuales
    X = X.dropna()
    y = target.iloc[-len(X):]  # Alinear target con features

    # 5. Split train/test (últimos horizon días para test)
    X_train, X_test = X[:-horizon], X[-horizon:]
    y_train, y_test = y[:-horizon], y[-horizon:]

    # 6. Entrenamiento del clasificador
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return X_test, y_test, y_pred, clf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
