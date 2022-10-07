import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

import pickle

from functions import create_rfm_dataset

import warnings

# warnings.filterwarnings(action="ignore")
warnings.filterwarnings(action="once")


# global simulation_results
# simulation_results = pd.DataFrame({})


def simulate_dataset(df, nb_days, nb_periods, output_path, experiment_nb):
    """
    Creates simulation datasets as global variables
    Saves simulation datasets as csv files
    :param df:
    :param nb_days:
    :param nb_periods:
    :param output_path:
    :param experiment_nb:
    :return: None
    :rtype: None
    """
    for index, i in enumerate(range(nb_periods, -1, -1)):  # A month = 15 days x 2 so 6 months needs 12 iterations

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
        globals()["exp_{}_rfm_T{}".format(str(experiment_nb), str(index))] = rfm_previous
        # 5) save csv
        rfm_previous.to_csv(output_path + "/rfm_T{}.csv".format(str(index)))  # , index=False)

        print("This dataset has {} unique clients".format(rfm_previous.shape[0]))
        # display(rfm_previous.head(2))
        # display(rfm_previous.info())


def evaluate_simulation(results, time, cls_init, cls_new):
    """

    :param results:
    :param time:
    :param cls_init:
    :param cls_new:
    :return:
    """
    print("ARI for T = {}".format(time))  # name Pandas Series
    ARI = adjusted_rand_score(cls_init, cls_new)

    results = pd.concat([results, pd.DataFrame({"T": [time],
                                                "ARI": [ARI]})], ignore_index=True)

    # results = results.sort_values(by=["ARI"], ascending=False)
    display(results)
    return results


def run_simulation(nb_periods, kmeans_cls_T0, results, experiment_nb, nb_clusters):
    """

    :param nb_periods:
    :param kmeans_cls_T0:
    :param results:
    :param experiment_nb:
    :return:
    """
    # we get the scaler at T0
    scaler_filename = './model/simulation/experiment_{}/scaler_T0.pkl'.format(str(experiment_nb))
    with open(scaler_filename, 'rb') as f:
        scaler_T0 = pickle.load(f)

    for t in range(1, nb_periods + 1):
        print("\n\n\nFor T =", t)
        # 1) we get the matrix X
        # X_T0 = pd.read_csv(output_path + "rfm_T0").drop("customer_unique_id", axis=1).copy()
        X = globals()["exp_{}_rfm_T{}".format(experiment_nb, str(t))]

        # 2) we scale the features
        # 2.1) New Standard Scaler
        X_std = X.copy()
        scaler = StandardScaler()
        X_std[X_std.columns] = scaler.fit_transform(X_std)
        print(X_std.shape)

        # 2.2) Using the scaler from T0
        X_std_scaler_T0 = X.copy()
        X_std_scaler_T0[X_std_scaler_T0.columns] = scaler_T0["scaler_T0"].transform(X_std_scaler_T0[X_std_scaler_T0.columns])

        # 3) Clustering
        # 3.1) new clustering
        print("We make a new clustering using that fits the new dataset.")
        kmeans_cls_new = KMeans(n_clusters=nb_clusters, verbose=0, random_state=0)
        kmeans_cls_new.fit(X_std)
        print(kmeans_cls_new)

        # 3.2) with initial clustering
        print("We predict a clustering using the clustering at T0 for the new dataset.")
        kmeans_cls_init = kmeans_cls_T0.predict(X_std_scaler_T0)

        # 4.1) evaluation ARI
        results = evaluate_simulation(results, time=t, cls_init=kmeans_cls_init,
                                      cls_new=kmeans_cls_new.labels_)

        # 4.2) evaluation Confusion Matrix
        matrix = confusion(y_true=kmeans_cls_init, y_pred=kmeans_cls_new.labels_)
        display(matrix)

    return results


def confusion(y_true, y_pred):
    """
    Displays a fancy confusion matrix
    :param y_test:
    :param y_pred:
    :return:
    """
    mat = confusion_matrix(y_true, y_pred) # a numpy array
    mat = pd.DataFrame(mat)
    mat.columns = [f"pred_{i}" for i in mat.columns]
    mat.index = [f"true_{i}" for i in mat.index]

    return mat


def display_ARI_nb_days(simulation_results, nb_days):
    """

    :param simulation_results:
    :param nb_days:
    :return:
    """
    sns.lineplot(data=simulation_results, x="T", y="ARI")
    # specifying horizontal line type
    plt.axhline(y=0.8, color='r', linestyle='-')

    plt.xlabel('The simulation period T, if T = 1, then we added {} days to T0.'.format(nb_days))
    plt.ylabel('ARI score')
    plt.title('ARI score as a function of the simulation period')
    plt.axis('tight')
