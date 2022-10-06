import pandas as pd
import numpy as np

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
            df[col + '_cluster'] = kmeans_clustering.predict(df[[col]])
            df = order_cluster(df, col + '_cluster', col, n_clusters, False)
        else:
            kmeans_clustering.fit(df[[col]])
            df[col + '_cluster'] = kmeans_clustering.predict(df[[col]])
            df = order_cluster(df, col + '_cluster', col, n_clusters, True)

    return df


# Applying the condition
## data_frame.loc[data_frame[feature] == old_value, feature] = new_value


def create_rfm_dataset(df, time_limit, review_score=False):
    """
    Converts "order_purchase_timestamp" to datetime object
    Returns a RFM dataset in which the Recency is computed based on the time_limit timestamp
    :param df: (DataFrame)
    :param time_limit: (datetime)
    :param review_score: (bool) if True, we add the mean review_score per customer
    :return:
    :rtype: (DataFrame)
    """
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'].astype(str), format='%Y/%m/%d')

    if not (review_score):
        dict_agg = {'order_purchase_timestamp': lambda x: (time_limit - x.max()).days,
                    'order_id': pd.Series.nunique,
                    'price': 'sum'  # ,'freight_value': 'sum'
                    }
    else:
        dict_agg = {'order_purchase_timestamp': lambda x: (time_limit - x.max()).days,
                    'order_id': pd.Series.nunique,
                    'price': 'sum',  # ,'freight_value': 'sum'
                    'review_score': 'mean'
                    }

    grouped_df = df.groupby('customer_unique_id').agg(
        dict_agg)  # grouped_df['total_payed'] = grouped_df['price'] + grouped_df['freight_value']
    # Rename columns
    rfm_dataset = grouped_df.rename(columns={'order_purchase_timestamp': 'Recency',
                                             'order_id': 'Frequency',
                                             'price': 'Monetary',
                                             'review_score': 'Review Score'})  # , inplace = True)
    return rfm_dataset
