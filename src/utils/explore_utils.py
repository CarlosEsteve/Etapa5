import pandas as pd
from tabulate import tabulate
from . import print_utils as pu

def explr(df):
    print(pu.headr('Columnas:'))
    print(df.columns.to_list())
    print(pu.headr('Numéricas'))
    print(df.describe())
    print(pu.headr('Categóricas'))
    print(df.describe(include=object))
