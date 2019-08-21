import sys
import numpy as np
import pandas as pd
from pandas.core import algorithms, base, generic, nanops, ops
from pandas.core.accessor import CachedAccessor
from pandas.core.arrays import ExtensionArray, SparseArray
from pandas.core.arrays.categorical import Categorical, CategoricalAccessor
import os

def ExamineData(df):
    """Prints various data charteristics, given x, where x is a pandas data frame
    """
    print("Data shape:", df.shape)
    print("\nColumn:", df.columns)
    print("\nData types", df.dtypes)
    print("\nDescribe data", df.describe())
    print("\nData ", df.head(2))
    print("\nSize of data", sys.getsizeof(df) /
          1000000, "MB")    # Get size of dataframes
    print("\nAre there any NULLS", np.sum(df.isnull()))


# [train set] =============================================================
def get_specific_colnames(dt):
    index = []
    for i, n in enumerate(dt.columns, start=0):
        if ('_index') in n:
            index.append(i)
    return index

def get_category_colnames(dt):
    index = []
    for i, n in enumerate(dt.columns, start=0):
        if ('building_material') in n:
            index.append(i)
        elif ('city') in n:
            index.append(i)
        elif ('building_type') in n:
            index.append(i)
        elif ('building_use') in n:
            index.append(i)
        elif ('parking_way') in n:
            index.append(i)
        elif ('total_floor') in n:
            index.append(i)
        elif ('town_uniq') in n:
            index.append(i)
        elif ('village_uniq') in n:
            index.append(i)
    return index

def get_nominal_column_index(dataframe):
    index = []
    for i, n in enumerate(dataframe.columns, start=0):
        if n == 'building_material':
            index.append(i)
        elif n == 'city':
            index.append(i)
        elif n == 'town_uniq':
            index.append(i)
        elif n == 'village_uniq':
            index.append(i)
        # elif n == 'total_floor':
        #     index.append(i)
        # elif n == 'txn_floor':
        #     index.append(i)
        elif n == 'building_type':
            index.append(i)
        elif n == 'building_use':
            index.append(i)
        elif n == 'parking_way':
            index.append(i)
    return index


def get_numerical_column_index_for_log(dataframe):
    index = []
    for i, n in enumerate(dataframe.columns, start=0):
        if n == 'land_area':
            index.append(i)
        elif n == 'building_area':
            index.append(i)
        elif n == 'town_population':
            index.append(i)
        elif n == 'town_area':
            index.append(i)
        elif n == 'town_population_density':
            index.append(i)
        elif n == 'total_price':
            index.append(i)
        elif('_MIN') in n:
            index.append(i)
    return index


def Normalize(dt):
    # s = 0.2 + 0.6 * (dt - dt.min()) / (dt.max() - dt.min())
    s = (dt - dt.min()) / (dt.max() - dt.min())
    Result = s
    return Result

# Normalize equivalent way

# x = kc_data_org.values  # returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# kc_data_nor = pd.DataFrame(x_scaled, columns=list(kc_data_org.columns.values))


def data_create_dummies(dataframe):
    # adjust unit of total_price
    dataframe['total_price'] = dataframe['total_price'] / 100000
    # adjust city-town-village to one column
    for col in ['city', 'town', 'village']:
        dataframe[col] = dataframe[col].astype('category').astype(str)
    dataframe['location'] = dataframe.city.str.cat(
        dataframe.town, sep='_').str.cat(dataframe.village, sep='_')
    dataframe.drop(
        dataframe[['city', 'town', 'village']], axis=1, inplace=True)
    # no-use columns
    dismiss_index = 3

    dataframe.drop(dataframe.columns[dismiss_index], axis=1, inplace=True)
    # nominal columns
    nominal_index = get_nominal_column_index(dataframe)
    nominal_colnames = list(dataframe.columns[nominal_index].values)
    dataframe.txn_floor = dataframe.txn_floor.fillna(999)
    for col in nominal_colnames:
        dataframe[col] = dataframe[col].astype('category')
    # print(dataframe[nominal_colnames].dtypes)
    dataframe = pd.get_dummies(dataframe,
                               columns=nominal_colnames,
                               prefix=nominal_colnames,
                               dtype=int
                               )
    return dataframe


def data_clean_for_train(dataframe):
    numerical_log10_index = get_numerical_column_index_for_log(dataframe)
    num_log10_colnames = list(dataframe.columns[numerical_log10_index].values)
    missing_row_index = dataframe.index[~dataframe.land_area.apply(
        bool)].tolist()
    dataframe.drop(missing_row_index, inplace=True)
    dataframe[num_log10_colnames] = dataframe[num_log10_colnames].apply(np.log10,
                                                                        axis=1,
                                                                        result_type='broadcast')
    Nanindex = dataframe.index[dataframe.village_income_median.isnull(
    )].tolist()
    dataframe.drop(Nanindex, inplace=True)
    dataframe.loc[:, 'txn_dt':'XIV_MIN'] = Normalize(
        dataframe.loc[:, 'txn_dt':'XIV_MIN'])
    return dataframe


# [test set] =============================================================

def get_y_max_and_min(dataframe):
    _max = dataframe.total_price.apply(np.log10).max()
    _min = dataframe.total_price.apply(np.log10).min()
    # _max = dataframe.total_price.max()
    # _min = dataframe.total_price.min()
    return _max, _min


def get_numerical_column_index_for_log_test(dataframe):
    index = []
    for i, n in enumerate(dataframe.columns, start=0):
        if n == 'land_area':
            index.append(i)
        elif n == 'building_area':
            index.append(i)
        elif n == 'town_population':
            index.append(i)
        elif n == 'town_area':
            index.append(i)
        elif n == 'town_population_density':
            index.append(i)
        elif('_MIN') in n:
            index.append(i)
    return index


def data_clean_for_test(dataframe):
    missing_row_index = dataframe.index[~dataframe.land_area.apply(
        bool)].tolist()
    dataframe.land_area = dataframe.land_area.replace(0, 1)
    numerical_log10_index = get_numerical_column_index_for_log_test(dataframe)
    num_log10_colnames = list(dataframe.columns[numerical_log10_index].values)
    dataframe[num_log10_colnames] = dataframe[num_log10_colnames].apply(np.log10,
                                                                        axis=1,
                                                                        result_type='broadcast')
    dataframe.loc[:, 'txn_dt': 'XIV_MIN'] = Normalize(
        dataframe.loc[:, 'txn_dt': 'XIV_MIN'])
    return dataframe


def check_filename_exist(filename):
    path = 'D:\\Programs\\python\\esan\\results'
    files = []
    # r=root, d=directories, f = files
    for r, f in os.walk(path):
        for file in f:
            if '.h5' in file:
                files.append(os.path.join(r, file))
    return print([f for f in files])
