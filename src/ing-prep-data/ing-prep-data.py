
import pandas as pd
import argparse
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def main(args):
    df = read_data(args.input_data)
    df2 = preprocess_data(df, args.equipo_input)
    output_df = df2.to_csv((Path(args.output_data) / "mantenimiento_df.csv"), index = False)


def read_data(uri):
    # 1. Cargar los datos
    data = pd.read_csv(uri)
    # 2. Inspeccionar las columnas
    print(data.columns)

    return data


def preprocess_data(data, equipo):
    data1 = data[data['ID_Equipo'] == equipo].copy()
    data1['Fecha'] = pd.to_datetime(data1['Fecha'], format='%Y-%m-%d')
    data1 = data1.set_index('Fecha')
    data1 = data1[~data1.index.duplicated(keep='first')]
    data1 = data1.asfreq('D')
    data1 = data1.sort_index()
    data1 = data1.drop(columns=['ID_Equipo','Ubicacion', 'Tipo_Equipo', 'Fabricante', 'Modelo', 'Potencia_kW', 'Horas_Recomendadas_Revision'])
    label_encoder = LabelEncoder()
    data1['Tipo_Mantenimiento'] = label_encoder.fit_transform(data1['Tipo_Mantenimiento'])
    print("Datos cargados y preprocesados:")
    print("Columnas definitivas:", data1.columns)
    print('ID_Equipo elegido:', equipo)

    return data1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", dest='input_data', type=str)
    parser.add_argument("--equipo_input", dest='equipo_input', type=int)
    parser.add_argument("--output_data", dest='output_data',type=str)
    args = parser.parse_args()

    return args


# run script
if __name__ == "__main__":
    args = parse_args()
    main(args)
