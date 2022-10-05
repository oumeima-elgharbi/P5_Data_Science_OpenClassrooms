import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from dateutil.relativedelta import relativedelta

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

        bp = sns.boxplot(data=df[variable])  # showfliers=False
        bp.set_title(variable)

    plt.tight_layout()
    plt.show()


def order_cluster(data_frame, col_cluster_name, col_name, nb_clusters, ascending):
    """
    Orders the names of the clusters for one column. So that the cluster 3 contains the best values.
    if nb_clusters = 4 : Cluster 0 < cluster 1 < cluster 2 < cluster 3

    :param data_frame: (DateFrame)
    :param col_cluster_name: (string)
    :param col_name: (string)
    :param nb_clusters: (int)
    :param ascending: (True or False)
    :return: a copy of data_frame with ordered clusters.
    :rtype: DataFrame
    """
    df = data_frame.copy()

    print("Before")
    display(df[[col_cluster_name, col_name]].groupby(col_cluster_name).describe())

    # step 1.1 : we get the minimum value for each cluster
    order = {}
    for nb_cls in range(nb_clusters):
        cls_min = df[df[col_cluster_name] == nb_cls].min()
        order[nb_cls] = cls_min[col_name]

    # step 1.2 : we make a list that contains the n° of the cluster and its min value
    cluster_number_min = [(order[key], key) for key in order.keys()]
    # True / increasing order
    if ascending:
        cluster_number_min.sort()  # print(l)
    # decreasing order
    else:
        cluster_number_min.sort(reverse=True)  # print(l)

    # step 2
    order_cluster = {}
    for nb_cls in range(nb_clusters):
        order_cluster[cluster_number_min[nb_cls][1]] = nb_cls  # print(order_cluster)

    # step 3 : we map the clusters with the correct order
    df[col_cluster_name] = df[col_cluster_name].map(order_cluster)

    # verification
    print("After")
    display(df[[col_cluster_name, col_name]].groupby(col_cluster_name).describe())

    return df


def apply_kmeans_per_column(data_frame, all_columns, kmeans_clustering, n_clusters):
    """

    :param data_frame: (DataFrame)
    :param all_columns: (list)
    :param kmeans_clustering: (K-Means)
    :param n_clusters: (int)
    :return:
    """
    df = data_frame.copy()

    for col in all_columns:
        if col == 'Recency':
            kmeans_clustering.fit(df[[col]])
            df[col +'_cluster'] = kmeans_clustering.predict(df[[col]])
            df = order_cluster(df, col +'_cluster', col, n_clusters, False)
        else:
            kmeans_clustering.fit(df[[col]])
            df[col +'_cluster'] = kmeans_clustering.predict(df[[col]])
            df = order_cluster(df, col +'_cluster', col, n_clusters, True)

    return df

# Applying the condition
## data_frame.loc[data_frame[feature] == old_value, feature] = new_value


def create_rfm_dataset(df, time_limit):
    """
    Converts "order_purchase_timestamp" to datetime object
    Returns a RFM dataset in which the Recency is computed based on the time_limit timestamp
    :param df: (DataFrame)
    :param time_limit: (datetime)
    :return:
    :rtype: (DataFrame)
    """
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'].astype(str), format='%Y/%m/%d')

    grouped_df = df.groupby('customer_unique_id').agg({'order_purchase_timestamp': lambda x: (time_limit - x.max()).days,
                                                       'order_id': pd.Series.nunique,
                                                       'price': 'sum',
                                                       })
    # Rename columns
    rfm_dataset = grouped_df.rename(columns = {'order_purchase_timestamp': 'Recency',
                                               'order_id': 'Frequency',
                                               'price': 'Monetary'}) #, inplace = True)
    return rfm_dataset


def simulate_dataset(df, nb_days, nb_periods, output_path):
    """
    Creates simulation datasets as global variables
    Saves simulation datasets as csv files
    :param df:
    :param nb_days:
    :param nb_periods:
    :param output_path:
    :return: None
    :rtype: None
    """
    for index, i in enumerate(range(nb_periods, -1, -1)): # A month = 15 days x 2 so 6 months needs 12 iterations

        # 1) Time limit
        time_limit = max(df.order_purchase_timestamp) + relativedelta(days=-nb_days * i)
        print("\n\n\nStep :", index + 1, "Maximum order purchase date :", time_limit, end='\n')

        # 2) filtering dataset based on time limit date
        data_previous = df.copy()
        filter_date = data_previous["order_purchase_timestamp"] <= time_limit
        data_previous = data_previous[filter_date]
        print("Verification of the filter :", max(data_previous.order_purchase_timestamp))

        # 3) Create a RFM dataset
        rfm_previous = create_rfm_dataset(data_previous, time_limit)

        # 4) we save the dataset in the global variables
        globals()["rfm_T" + str(index)] = rfm_previous
        # 5) save csv
        rfm_previous.to_csv(output_path + "rfm_T" + str(index))#, index=False)

        print("This dataset has {} unique clients".format(rfm_previous.shape[0]))
        #display(rfm_previous.head(2))
        #display(rfm_previous.info())