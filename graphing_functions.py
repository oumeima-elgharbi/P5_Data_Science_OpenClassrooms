import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale


import warnings

# warnings.filterwarnings(action="ignore")
warnings.filterwarnings(action="once")


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
    plt.figure(figsize=(12, 5))
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

    ax = fig.add_subplot(121)  # 1 en ordonnée / 2 en abcs / celle là la premiere
    ax.scatter(x=X_pca[:, 0], y=X_pca[:, 1], c=cls.labels_)  # colorier en fct etiquette deu clusterning
    plt.title("Visualizing clusters with PCA")

    ax = fig.add_subplot(122)  # 1 en ordonnée / 2 en abcs / celle là la premiere
    ax.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], c=cls.labels_)  # palette=sns.color_palette("hls", n_colors=num_clusters)
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
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=cls.labels_,
        ax=ax2
    )


def display_barplot_total_customers(grouped_df):
    plt.figure(figsize=(12, 5))

    # Percentage
    ax = plt.subplot(1, 2, 1)
    plt.title('Distribution of the number of customers per cluster (in percent)')
    sns.barplot(x=grouped_df.index,
                y=grouped_df['Nb customers'] / grouped_df["Nb customers"].sum() * 100,
                ax=ax)  # the index of the df represents the clusters
    plt.xlabel('Cluster')
    plt.ylabel("Number of customers in %")

    # Total number
    ax = plt.subplot(1, 2, 2)
    plt.title('Distribution of the number of customers per cluster)')
    sns.barplot(x=grouped_df.index,
                y=grouped_df['Nb customers'],
                ax=ax)  # the index of the df represents the clusters
    plt.xlabel('Cluster')
    plt.ylabel("Number of customers")


def display_barplot_avg_per_feature(df_grouped, all_features):
    plt.figure(figsize=(20, 5))

    n = len(all_features)

    for i, feature in enumerate(all_features):
        ax = plt.subplot(1, n, i + 1)
        plt.title('Distribution of the average {} per cluster'.format(feature))
        sns.barplot(x=df_grouped.index,
                    y=df_grouped['Avg {}'.format(feature)],
                    ax=ax)  # the index of the df represents the clusters
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
        # feature= pd.Series(features_to_plot)
        bp = sns.boxplot(data=df, x=cluster_column_name, y=feature, ax=ax)

        bp.set_title("Distribution of {} per cluster".format(feature))
        bp.set_xlabel("Cluster")
        bp.set_ylabel("")
        plt.tight_layout()
        # plt.show()


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
        # plt.show()


def display_boxplot_per_feature_per_cluster(df, feature, cluster_column_name):
    """

    :param df:
    :return:
    """
    # we create a sorted list of the cluster numbers
    list_cluster = sorted(df[cluster_column_name].unique().tolist())
    n = len(list_cluster)

    fig = plt.figure(figsize=(10, 4))

    for i, num_cls in enumerate(list_cluster):  # sorted so that we display cluster 0 then  etc
        ax = plt.subplot(1, n, i + 1)

        # we select all the customers that belong to the same cluster
        df_cluster = df[df[cluster_column_name] == num_cls]
        bp = sns.boxplot(data=df_cluster, y=feature, ax=ax)

        bp.set_title("{} for the cluster {}".format(feature, num_cls), fontsize=10)
        bp.set_xlabel("Cluster : " + str(num_cls))
        bp.set_ylabel("")
        plt.tight_layout()


def display_3D_rfm(df, labels):
    """

    :param df:
    :param labels:
    :return:
    """
    fig = plt.figure(figsize=(25, 25))
    plt.clf()

    ax = fig.add_subplot(111, projection='3d',  # ,
                         elev=48,
                         azim=134)
    plt.cla()
    ax.scatter(df["Recency"], df["Frequency"], df["Monetary"],
               c=labels,
               s=200,
               cmap='spring',
               alpha=0.5,
               edgecolor='darkgrey'
               )

    ax.set_title("3D representation of the RFM features per cluster")
    ax.set_xlabel('Recency', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.set_zlabel('Monetary', fontsize=16)


def display_radar_normalized(grouped_df, list_cluster):
    """

    :param grouped_df:
    :param list_cluster:
    :return:
    """
    # 1) we scale the grouped df so that we can have a better display on the radar plot.
    # we did try MinMaxScaler and Scale : the display with Scale was better.
    print("we scale the grouped df so that we can have a better display on the radar plot")
    df_norm = grouped_df.copy()
    df_norm[df_norm.columns] = scale(df_norm)

    # using a MinMaxScaler :
    # scaler = MinMaxScaler().fit(df_norm)
    # df_norm[df_norm.columns] = scaler.transform(df_norm)

    # 2) making the radar plot
    categories = grouped_df.columns.tolist()
    # categories.remove("Nb customers")
    fig = go.Figure()

    # for each cluster, we take the Series associated to it and transform it into a list
    for num_cls in list_cluster:
        fig.add_trace(go.Scatterpolar(
            r=df_norm[df_norm.index == num_cls].values.tolist()[0],  # [1:], # we have a list of a list so [0]
            theta=categories,
            fill='toself',
            name='Cluster {}'.format(num_cls)
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True  # ,
                #  range=[0, 5]
            )),
        showlegend=False
    )

    fig.show()