import pandas as pd
import numpy as np
import os
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_validate
from skforecast.exceptions import LongTrainingWarning

# Tratamiento de datos
# ==============================================================================
import shap
from skforecast.utils import load_forecaster
from skforecast.utils import save_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import TimeSeriesFold
from skforecast.direct import ForecasterDirect
from skforecast.recursive import ForecasterRecursive
import skforecast
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import sklearn
from skforecast.datasets import fetch_dataset

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 10

# Modelado y Forecasting
# ==============================================================================


# Configuración warnings
# ==============================================================================
warnings.filterwarnings('once')

color = '\033[1m\033[38;5;208m'
print(f"{color}Versión skforecast: {skforecast.__version__}")
print(f"{color}Versión scikit-learn: {sklearn.__version__}")
print(f"{color}Versión pandas: {pd.__version__}")
print(f"{color}Versión numpy: {np.__version__}")

# Formato de los prints
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def headr(text):
    return ('\n'+color.UNDERLINE + text + color.END+'\n')


def preprocesado(ruta, equipo):
    mantenimiento_df = pd.read_csv(ruta + 'Etapa4/output/merge_df.csv')
    mantenimiento_df_1 = mantenimiento_df[mantenimiento_df['ID_Equipo'] == equipo].copy()
    mantenimiento_df_1['Fecha'] = pd.to_datetime(mantenimiento_df_1['Fecha'], format='%Y-%m-%d')
    mantenimiento_df_1 = mantenimiento_df_1.set_index('Fecha')
    mantenimiento_df_1 = mantenimiento_df_1[~mantenimiento_df_1.index.duplicated(keep='first')]
    mantenimiento_df_1 = mantenimiento_df_1.asfreq('D')
    mantenimiento_df_1 = mantenimiento_df_1.sort_index()
    print("\nSumatorio de filas duplicadas en Dataframe:", mantenimiento_df_1.duplicated().sum())
    print(f'Número de filas con missing values: {mantenimiento_df_1.isnull().any(axis=1).mean()}')
    fecha_inicio = mantenimiento_df_1.index.min()
    fecha_fin = mantenimiento_df_1.index.max()
    date_range_completo = pd.date_range(start=fecha_inicio, end=fecha_fin, freq=mantenimiento_df_1.index.freq)
    print(f"Índice completo: {(mantenimiento_df_1.index == date_range_completo).all()}")
    print("Tratamiento Dataframe finalizado")
    print('\n''Dataframe limpio con índice de fecha para el ID_Euipo:', equipo)
    print(tabulate(mantenimiento_df_1.head(5), headers='keys', tablefmt='psql'))
    mantenimiento_df_1 = mantenimiento_df_1.drop(columns=['ID_Equipo','Ubicacion', 'Tipo_Equipo', 'Fabricante', 'Modelo', 'Potencia_kW', 'Horas_Recomendadas_Revision'])
    label_encoder = LabelEncoder()
    mantenimiento_df_1['Tipo_Mantenimiento'] = label_encoder.fit_transform(mantenimiento_df_1['Tipo_Mantenimiento'])
    return mantenimiento_df_1

def separacion_train_test(mantenimiento_df_1, steps, equipo):
    datos_train = mantenimiento_df_1[:-steps]
    datos_test  = mantenimiento_df_1[-steps:]
    print('\nSeparación Dataframe para su entrenamiento:')
    print(f"Fechas train : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
    print(f"Fechas test  : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")
    fig, ax = plt.subplots(figsize=(6, 2.5))
    datos_train['Horas_Operativas'].plot(ax=ax, label='train')
    datos_test['Horas_Operativas'].plot(ax=ax, label='test')
    ax.legend()
    ax.set_title(f"Separación Train/Test para el Equipo {equipo}")
    return datos_train, datos_test

def crear_entrenar_forecaster(datos_train):
    forecaster = ForecasterRecursive(
        regressor=RandomForestRegressor(n_estimators= 100, random_state=123),
        lags=470
    )
    exogs_train = datos_train.drop(columns=['Horas_Operativas'])
    forecaster.fit(y=datos_train['Horas_Operativas'], exog=exogs_train)
    return forecaster

def predicciones_forecaster(forecaster, datos_train, datos_test, steps, equipo):
    exogs_test = datos_test.drop(columns=['Horas_Operativas'])
    predicciones = forecaster.predict(steps=steps, exog=exogs_test)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    datos_train['Horas_Operativas'].plot(ax=ax, label='train')
    datos_test['Horas_Operativas'].plot(ax=ax, label='test')
    predicciones.plot(ax=ax, label='predicciones')
    ax.legend()
    ax.set_title(f"Predicciones para el Equipo {equipo}")
    error_mse = mean_squared_error(y_true=datos_test['Horas_Operativas'], y_pred=predicciones)
    varianza = datos_test['Horas_Operativas'].var()
    print(f'\nMétricas resultados predicciones:')
    print(f"Error de test (mse): {error_mse}")
    print(f"Varianza de los valores reales: {varianza}")
    return predicciones

def backtesting_forecaster_func(forecaster, mantenimiento_df_1, steps, equipo):
    cv = TimeSeriesFold(
        steps=steps,
        initial_train_size=int(len(mantenimiento_df_1) * 0.5),
        fixed_train_size=False,
        refit=True,
    )
    metrica, predicciones_backtest = backtesting_forecaster(
        forecaster=forecaster,
        y=mantenimiento_df_1['Horas_Operativas'],
        cv=cv,
        metric='mean_squared_error',
        verbose=False
    )
    fig, ax = plt.subplots(figsize=(6, 2.5))
    mantenimiento_df_1.loc[predicciones_backtest.index, 'Horas_Operativas'].plot(ax=ax, label='test')
    predicciones_backtest.plot(ax=ax, label='predicciones')
    ax.legend()
    ax.set_title(f"Backtesting para el Equipo {equipo}")
    return metrica

def orquestador_horas(ruta, equipo, steps, backtesting=False):
    print("Tratamiento Dataframe iniciado:")
    mantenimiento_df_1 = preprocesado(ruta, equipo)
    datos_train, datos_test = separacion_train_test(mantenimiento_df_1, steps, equipo)
    forecaster = crear_entrenar_forecaster(datos_train)
    predicciones = predicciones_forecaster(forecaster, datos_train, datos_test, steps, equipo)
    if backtesting:
        metrica = backtesting_forecaster_func(forecaster, mantenimiento_df_1, steps, equipo)
        print('\nValores estimados:')
        print(predicciones)
        return predicciones, metrica
    else:
        print('\nValores estimados:')
        print(predicciones)
        return predicciones
    return
