import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import csv
import h5py
import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model

def mape(preds, dtrain):
    labels = dtrain.get_label()
    return('mape', np.mean(np.abs((preds - labels) / (labels))*100))

    # 自定義一個 hitrate


def hitrate(preds, dtrain):
    labels = dtrain.get_label()
    return('hitrate', np.sum(np.where((np.abs((preds-labels) / (labels))) <= 0.1, 1, 0))/len(preds))


def train_validate_test_split(df, train_part=.6, validate_part=.2, test_part=.2, seed=None):
    np.random.seed(seed)
    total_size = train_part + validate_part + test_part
    train_percent = train_part / total_size
    validate_percent = validate_part / total_size
    test_percent = test_part / total_size
    perm = np.random.permutation(df.index)
    m = len(df)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = perm[:train_end]
    validate = perm[train_end:validate_end]
    test = perm[validate_end:]
    return train, validate, test


def train_validate_split(df, train_part=.6, validate_part=.2, seed=None):
    np.random.seed(seed)
    total_size = train_part + validate_part
    train_percent = train_part / total_size
    validate_percent = validate_part / total_size
    perm = np.random.permutation(df.index)
    m = len(df)
    train_end = int(train_percent * m)
    # validate_end = int(validate_percent * m) + train_end
    train = perm[:train_end]
    validate = perm[train_end:]
    # test = perm[validate_end:]
    return train, validate


def norm_stats(df1, df2):
    import numpy as np
    dfs = df1.append(df2)
    minimum = np.min(dfs)
    maximum = np.max(dfs)
    mu = np.mean(dfs)
    sigma = np.std(dfs)
    return (minimum, maximum, mu, sigma)


def z_score(col, stats):
    import pandas as pd  # 和上面建立模型一樣，標準化流程
    epsilon = 0.00000001
    m, M, mu, s = stats
    df = pd.DataFrame()
    for c in col.columns:
        df[c] = (col[c] - mu[c]) / (s[c] + epsilon)  # 分母加epsilon避免為0
    return df


def basic_model_1(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(100, activation="tanh", input_shape=(x_size,)))
    t_model.add(Dense(50, activation="relu"))
    t_model.add(Dense(y_size))
    print(t_model.summary())
    t_model.compile(loss='mean_squared_error',
                    optimizer=Adam(),
                    metrics=[metrics.mae])
    return(t_model)


def basic_model_2(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(100, activation="tanh", input_shape=(x_size,)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(50, activation="relu"))
    t_model.add(Dense(20, activation="relu"))
    t_model.add(Dense(y_size))
    print(t_model.summary())
    t_model.compile(loss='mean_squared_error',
                    optimizer=Adam(),
                    metrics=[metrics.mae])
    return(t_model)


def basic_model_3(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(80, activation="tanh", kernel_initializer='normal', input_shape=(x_size,)))
    t_model.add(Dropout(0.2))
    t_model.add(Dense(120, activation="relu", kernel_initializer='normal',
                      kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(20, activation="relu", kernel_initializer='normal',
                      kernel_regularizer=regularizers.l1_l2(0.01), bias_regularizer=regularizers.l1_l2(0.01)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(10, activation="relu", kernel_initializer='normal'))
    t_model.add(Dropout(0.0))
    t_model.add(Dense(y_size))
    t_model.compile(
        loss='mean_squared_error',
        optimizer='nadam',
        metrics=[metrics.mae])
    return(t_model)


def basic_model_3_sd(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(600, activation="tanh", kernel_initializer='normal', input_shape=(x_size,)))
    t_model.add(Dropout(0.2))
    t_model.add(Dense(700, activation="relu", kernel_initializer='normal',
                      kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(200, activation="relu", kernel_initializer='normal',
                      kernel_regularizer=regularizers.l1_l2(0.01), bias_regularizer=regularizers.l1_l2(0.01)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(100, activation="relu", kernel_initializer='normal'))
    t_model.add(Dropout(0.0))
    t_model.add(Dense(y_size))
    t_model.compile(
        loss='mean_squared_error',
        optimizer='nadam',
        metrics=[metrics.mae])
    return(t_model)


def basic_model_3_pw(x_size, y_size):

    t_model = Sequential()
    t_model.add(Dense(600, activation="tanh", kernel_initializer='normal', input_shape=(x_size,)))
    t_model.add(Dropout(0.2))
    t_model.add(Dense(700, activation="relu", kernel_initializer='normal',
                      kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(700, activation="relu", kernel_initializer='normal',
                      kernel_regularizer=regularizers.l1_l2(0.01), bias_regularizer=regularizers.l1_l2(0.01)))
    t_model.add(Dropout(0))
    t_model.add(Dense(300, activation="relu", kernel_initializer='normal',
                      kernel_regularizer=regularizers.l1_l2(0.01), bias_regularizer=regularizers.l1_l2(0.01)))
    t_model.add(Dropout(0))
    t_model.add(Dense(300, activation="relu", kernel_initializer='normal'))
    t_model.add(Dropout(0))
    t_model.add(Dense(200, activation="relu", kernel_initializer='normal'))
    t_model.add(Dropout(0))
    t_model.add(Dense(200, activation="relu", kernel_initializer='normal'))
    t_model.add(Dropout(0.0))
    t_model.add(Dense(y_size))
    t_model.compile(
        loss='mean_squared_error',
        optimizer='nadam',
        metrics=[metrics.mae])
    return(t_model)
