import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

from os import listdir
from os.path import isfile, join

# to compute time of pipeline
from time import time, strftime, gmtime

import warnings

# warnings.filterwarnings(action="ignore")
warnings.filterwarnings(action="once")


def load_data(path, filename):
    """
    Step 0)
    :param path:
    :param filename: (string)
    :return:
    """
    print("___Loading raw dataset___")

    # Load raw data
    dataset_file = "{}{}".format(path, filename)
    dataset = pd.read_csv(dataset_file)

    print("Initial shape :", dataset.shape)
    return dataset


def density(df, lines=7, cols=4):
    """
    Input : dataframe, lignes, colonnes
    Output : grille des courbes de densités des variables numériques du dataframe
    """
    print("___Density distribution___")
    df = df.select_dtypes(include='number').copy()

    fig, ax = plt.subplots(lines, cols, figsize=(min(15, cols * 3), lines * 2))

    for i, val in enumerate(df.columns.tolist()):
        bp = sns.distplot(df[val], hist=False, ax=ax[i // cols, i % cols], kde_kws={'shade': True})
        bp.set_title("skewness : " + str(round(df[val].skew(), 1)), fontsize=12)
        bp.set_yticks([])
        imax = i

    for i in range(imax + 1, lines * cols):
        ax[i // cols, i % cols].axis('off')

    plt.tight_layout()
    plt.show()


def density_histplot(df, lines=7, cols=4):
    """
    Input : dataframe, lignes, colonnes
    Output : grille des courbes de densités des variables numériques du dataframe
    """
    print("___Density distribution___")
    df = df.select_dtypes(include='number').copy()

    fig, ax = plt.subplots(lines, cols, figsize=(min(15, cols * 3), lines * 2))

    for i, val in enumerate(df.columns.tolist()):
        bp = sns.histplot(df[val], ax=ax[i // cols, i % cols], kde=True)  # kde_kws={'shade': True})
        bp.set_title("skewness : " + str(round(df[val].skew(), 1)), fontsize=12)
        bp.set_yticks([])
        imax = i

    for i in range(imax + 1, lines * cols):
        ax[i // cols, i % cols].axis('off')

    plt.tight_layout()
    plt.show()


def correlation_matrix(df, width=8, height=6):
    # we create a dataframe with all the numerical variables
    numeric_columns = df.select_dtypes(include='number').columns.tolist()

    df_to_corr = df[numeric_columns]

    # we assign the type float to all the values of the matrix
    df_to_corr = df_to_corr.astype(float)
    corr = df_to_corr.corr(method='pearson')

    plt.figure(figsize=(width, height))

    # sns.heatmap(corr, annot=True, vmin=-1, cmap='coolwarm')
    sns.heatmap(corr, center=0, cmap=sns.color_palette("RdBu_r", 7), linewidths=1,
                annot=True, annot_kws={"size": 9}, fmt=".02f")

    plt.title('Correlation matrix - Pearson', fontsize=18)
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12)
    plt.show()


def display_boxplot(df, width=8, height=6):
    """

    :return:
    """
    numeric_columns = df.select_dtypes(include='number').columns.tolist()

    fig = plt.figure(figsize=(width, height))
    for i, variable in enumerate(df[numeric_columns].columns.tolist()):
        position = int('13{}'.format(i + 1))
        ax = fig.add_subplot(position)

        bp = sns.boxplot(data=df[variable]) # showfliers=False
        bp.set_title(variable)

    plt.tight_layout()
    plt.show()
