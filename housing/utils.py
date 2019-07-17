import os
import pandas as pd
from pprint import pprint

from housing.const import HOUSING_PATH


def get_housing_data():
    csv_path = os.path.join(HOUSING_PATH, "housing.csv")
    return pd.read_csv(csv_path)


def print_full(x):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pprint(x)
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')
