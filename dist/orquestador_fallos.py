import pandas as pd
import numpy as np
import warnings
from tabulate import tabulate
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import sklearn

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 10

# Configuración warnings
# ==============================================================================
warnings.filterwarnings('once')

color = '\033[1m\033[38;5;208m'
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

def load_data(ruta, equipo, steps):
    mantenimiento_df = pd.read_csv(ruta + 'Etapa4/output/merge_df.csv')
    mantenimiento_df_1 = mantenimiento_df[mantenimiento_df['ID_Equipo'] == equipo].copy()
    mantenimiento_df_1['Fecha'] = pd.to_datetime(mantenimiento_df_1['Fecha'], format='%Y-%m-%d')
    mantenimiento_df_1 = mantenimiento_df_1.set_index('Fecha')
    mantenimiento_df_1 = mantenimiento_df_1[~mantenimiento_df_1.index.duplicated(keep='first')]
    mantenimiento_df_1 = mantenimiento_df_1.asfreq('D')
    mantenimiento_df_1 = mantenimiento_df_1.sort_index()
    mantenimiento_df_1 = mantenimiento_df_1.drop(columns=['ID_Equipo','Ubicacion', 'Tipo_Equipo', 'Fabricante', 'Modelo', 'Potencia_kW', 'Horas_Recomendadas_Revision'])
    label_encoder = LabelEncoder()
    mantenimiento_df_1['Tipo_Mantenimiento'] = label_encoder.fit_transform(mantenimiento_df_1['Tipo_Mantenimiento'])
    print("Datos cargados y preprocesados:")
    print('ID_Equipo:', equipo)
    print('steps:', steps)
    return mantenimiento_df_1

def split_data(mantenimiento_df_1, steps):
    print("Dividiendo datos en conjuntos de entrenamiento y prueba...")
    datos_train = mantenimiento_df_1[:-steps]
    datos_test  = mantenimiento_df_1[-steps:]
    data = mantenimiento_df_1['Horas_Operativas']
    target = mantenimiento_df_1['Tipo_Mantenimiento']
    print(f"Fechas train : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
    print(f"Fechas test  : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")
    return datos_train, datos_test, data, target

def create_forecaster(data):
    forecaster = RandomForestRegressor(n_estimators=100, random_state=123)
    forecaster.fit(np.arange(len(data)).reshape(-1, 1), data)
    print("Modelo de pronóstico creado y entrenado.")
    return forecaster

def predict_forecaster(forecaster, data, target):
    predicciones = forecaster.predict(np.arange(len(target)).reshape(-1, 1))
    residuals = data.values - predicciones
    return predicciones, residuals

def create_features_w_residuals(data, residuals, lags):
    print("Creando características basadas en los residuos...")
    features = pd.concat([data.shift(lag).rename(f'lag_{lag}') for lag in range(1, lags + 1)], axis=1)
    features['residuals'] = residuals
    features.dropna(inplace=True)
    print(tabulate(features.head(5), headers='keys', tablefmt='psql'))
    return features
def split_train_test(X, y, steps):
    print("Dividiendo características y objetivo en conjuntos de entrenamiento y prueba...")
    X_train, X_test = X[:-steps], X[-steps:]
    y_train, y_test = y[:-steps], y[-steps:]
    return X_train, X_test, y_train, y_test

def fit_rf_model(X_train, y_train):
    print("Ajustando el modelo de bosque aleatorio...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def predict_rf_model(rf_model, X_test, equipo):
    print("Realizando predicciones con el modelo de bosque aleatorio...")
    y_pred = rf_model.predict(X_test)
    print('\nPredicciones realizadas para el ID_Equipo:', equipo)
    print('0 FALLO 1 NO FALLO')
    print(y_pred)
    return y_pred

def evaluate_model(y_test, y_pred):
    print("\nEvaluando el modelo...")
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc}')
    return acc

def orchestrator_fallos(ruta, equipo, steps):
    mantenimiento_df_1 = load_data(ruta, equipo, steps)
    datos_train, datos_test, data, target = split_data(mantenimiento_df_1, steps)
    forecaster = create_forecaster(data)
    predicciones, residuals = predict_forecaster(forecaster, data, target)
    X = create_features_w_residuals(data, residuals, lags=470)
    y = target.iloc[len(target) - len(X):]
    X_train, X_test, y_train, y_test = split_train_test(X, y, steps)
    rf_model = fit_rf_model(X_train, y_train)
    y_pred = predict_rf_model(rf_model, X_test, equipo)
    acc = evaluate_model(y_test, y_pred)
    return acc
