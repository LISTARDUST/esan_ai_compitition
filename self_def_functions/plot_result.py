import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_hist(h, xsize=6, ysize=10):

    # Prepare plotting
    fig_size = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = [xsize, ysize]
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True)

    # summarize history for MAE
    plt.subplot(211)
    plt.plot(h['mean_absolute_error'])
    plt.plot(h['val_mean_absolute_error'])
    plt.title('Training vs Validation MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot it all in IPython (non-interactive)
    plt.draw()
    plt.show()

    return


def check_variables_distribution(dt):

    names = dt.columns.values
    # print(names)
    # print(len(names))
    m = 6
    n = len(names) // m
    # print(n)
    plot_data = dt.dropna()
    # plot_data.plot(kind='box', subplots=True, layout=(n, m), sharex=False)
    citylist = [dt['city'].unique().value]
    print(citylist)
    pal = dict()

    g = sns.FacetGrid(plot_data, col='city', height=4, aspect=1,
                      hue='town', palette='Accent', hue_kws={'marker': ['D', 's']})
    g.map(plt.scatter, 'town', 'total_price', alpha=0.4)
    # sns.set(font_scale=1.5)
    # g = sns.PairGrid(plot_data[:, ], hue='city')
    # g.map_diag(plt.hist)
    # g.map_offdiag(plt.scatter)
    # g.add_legend()
    plt.show()


# print('[0] load data ...')
# # tStart = time.time()  # 計時開始
# org_train = pd.read_csv("D:\\DT\\esan\\original\\train.csv")
# org_test = pd.read_csv('D:\\DT\\esan\\original\\test.csv')
# kc_data_org = org_train.append(org_test, sort=False, ignore_index=True)
# check_variables_distribution(kc_data_org)
