import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    n = len(numeric_columns)

    fig = plt.figure(figsize=(width, height))
    for i, variable in enumerate(df[numeric_columns].columns.tolist()):
        position = int('1{}{}'.format(n, i + 1))
        ax = fig.add_subplot(position)

        bp = sns.boxplot(data=df[variable], ax=ax)  # showfliers=False
        bp.set_title(variable)

    plt.tight_layout()
    plt.show()


def display_pca_variance_cumsum(pca_fitted):
    """

    :param pca_fitted:
    :return:
    """
    plt.figure(figsize=(12,5))
    plt.title('PCA : Cumulated sum of explained variance as a function of the number of components')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulated variance explained')

    plt.plot(np.arange(start=1, stop=pca_fitted.n_components + 1),
             np.cumsum(pca_fitted.explained_variance_ratio_))


def display_pca_components(pca_fit, X_norm):
    """

    :param pca_fit:
    :param X_norm:
    :return:
    """
    pcs = pca_fit.components_

    for i, (x, y) in enumerate(zip(pcs[0, :], pcs[1, :])):
        # Afficher un segment de l'origine au point (x, y)
        plt.plot([0, x], [0, y], color='k')
        # Afficher le nom (data.columns[i]) de la performance
        plt.text(x, y, X_norm.columns[i], fontsize='14')

        # Afficher une ligne horizontale y=0
        plt.plot([-0.7, 0.7], [0, 0], color='grey', ls='--')

        # Afficher une ligne verticale x=0
        plt.plot([0, 0], [-0.7, 0.7], color='grey', ls='--')

        plt.xlim([-0.7, 0.7])
        plt.ylim([-0.7, 0.7])


def display_pca_tsne(X_pca, X_tsne, cls):
    """

    :param X_pca:
    :param X_tsne:
    :param cls:
    :return:
    """
    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121) # 1 en ordonnée / 2 en abcs / celle là la premiere
    ax.scatter(x=X_pca[:, 0], y=X_pca[:, 1], c=cls.labels_) # colorier en fct etiquette deu clusterning
    plt.title("Visualizing clusters with PCA")

    ax = fig.add_subplot(122) # 1 en ordonnée / 2 en abcs / celle là la premiere
    ax.scatter(x=X_tsne[:,0], y=X_tsne[:,1], c=cls.labels_) # palette=sns.color_palette("hls", n_colors=num_clusters)
    plt.title('Principal Components projection with t-SNE')


def display_pca_tsne_sns(X_pca, X_tsne, cls):
    """

    :param X_pca:
    :param X_tsne:
    :param cls:
    :return:
    """
    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=cls.labels_,
        ax=ax1
    )

    ax2 = plt.subplot(1, 2, 2)
    sns.scatterplot(
        x=X_tsne[:,0], y=X_tsne[:,1],
        hue=cls.labels_,
        ax=ax2
    )


def display_barplot_avg_per_feature(df_grouped, all_features):

    plt.figure(figsize=(20, 5))

    n = len(all_features)

    for i, feature in enumerate(all_features):
        ax = plt.subplot(1, n, i + 1)
        plt.title('Distribution of the average {} per cluster'.format(feature))
        sns.barplot(x=df_grouped.index,
                    y=df_grouped['Avg {}'.format(feature)],
                    ax=ax) # the index of the df represents the clusters
        plt.xlabel('Cluster')
        plt.ylabel("Average {} per cluster".format(feature))



def display_boxplot_per_feature(df, all_features, cluster_column_name):
    """

    :param df:
    :param all_features:
    :param cluster_column_name:
    :return:
    """
    n = len(all_features)
    fig = plt.figure(figsize=(15, 4))

    for i, feature in enumerate(all_features):
        ax = plt.subplot(1, n, i + 1)
        # we select all the customers that belong to the same cluster
        #feature= pd.Series(features_to_plot)
        bp = sns.boxplot(data=df, x=cluster_column_name, y=feature, ax=ax)

        bp.set_title("Distribution of {} per cluster".format(feature))
        bp.set_xlabel("Cluster")
        bp.set_ylabel("")
        plt.tight_layout()
        #plt.show()



def display_boxplot_per_cluster(df, cluster_column_name):
    """

    :param df:
    :param cluster_column_name:
    :param cluster_nb:
    :return:
    """
    list_cluster = sorted(df[cluster_column_name].unique().tolist())
    n = len(list_cluster)
    fig = plt.figure(figsize=(18, 4))

    for i, cls_nb in enumerate(list_cluster):
        ax = plt.subplot(1, n, i + 1)
        # we select all the customers that belong to the same cluster
        df_cluster = df[df[cluster_column_name] == cls_nb]
        bp = sns.boxplot(data=df_cluster, ax=ax)

        bp.set_title("For the cluster {}".format(str(cls_nb)))
        bp.set_xlabel("Features")
        bp.set_ylabel("")
        plt.tight_layout()
        #plt.show()


def display_boxplot_per_feature_per_cluster(df, feature, cluster_column_name):
    """

    :param df:
    :return:
    """
    # we create a sorted list of the cluster numbers
    list_cluster = sorted(df[cluster_column_name].unique().tolist())
    n = len(list_cluster)

    fig = plt.figure(figsize=(10, 4))

    for i, num_cls in enumerate(list_cluster): # sorted so that we display cluster 0 then  etc
        ax = plt.subplot(1, n, i + 1)

        # we select all the customers that belong to the same cluster
        df_cluster = df[df[cluster_column_name] == num_cls]
        bp = sns.boxplot(data=df_cluster, y=feature, ax=ax)

        bp.set_title("{} for the cluster {}".format(feature, num_cls), fontsize=10)
        bp.set_xlabel("Cluster : " + str(num_cls))
        bp.set_ylabel("")
        plt.tight_layout()






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
                                             'price': 'Monetary'})  # , inplace = True)
    return rfm_dataset
