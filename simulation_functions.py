import pandas as pd

from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

from functions import create_rfm_dataset

# to compute time of pipeline
from time import time, strftime, gmtime

import warnings

# warnings.filterwarnings(action="ignore")
warnings.filterwarnings(action="once")

#global simulation_results
#simulation_results = pd.DataFrame({})


def simulate_dataset(df, nb_days, nb_periods, output_path, experiment_nb):
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

    :param simulation_results:
    :param time:
    :param cls_init:  (np array)
    :param cls_new:  (np array)
    :return:
    """
    print("ARI for T = {}".format(time))  # name Pandas Series
    ARI = adjusted_rand_score(cls_init, cls_new)

    results = pd.concat([results, pd.DataFrame({"T": [time],
                                                "ARI": [ARI]})], ignore_index=True)

    #results = results.sort_values(by=["ARI"], ascending=False)
    display(results)
    return results


def run_simulation(nb_periods, kmeans_cls_T0, results, experiment_nb):
    """

    :param nb_periods:
    :param kmeans_cls_T0:
    :return:
    """

    for t in range(1, nb_periods + 1):
        print("\n\n\nFor T =", t)
        # 1) we get the matrix X
        # X_T0 = pd.read_csv(output_path + "rfm_T0").drop("customer_unique_id", axis=1).copy()
        X = globals()["exp_{}_rfm_T{}".format(experiment_nb, str(t))]

        # 2) we scale the features
        X_std = X.copy()
        scaler = StandardScaler()
        X_std[X_std.columns] = scaler.fit_transform(X_std)
        print(X_std.shape)

        # 3) Clustering
        # new clustering
        kmeans_cls_new = KMeans(n_clusters=5, verbose=0, random_state=0)
        kmeans_cls_new.fit(X_std)
        print(kmeans_cls_new)

        # with initial clustering
        kmeans_cls_init = kmeans_cls_T0.predict(X_std)

        # 3) evaluation ARI
        results = evaluate_simulation(results, time=t, cls_init=kmeans_cls_init,
                                                 cls_new=kmeans_cls_new.labels_)
    return results
