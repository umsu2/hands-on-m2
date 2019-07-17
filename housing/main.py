from pprint import pprint
from matplotlib import pyplot as plt
from housing.utils import get_housing_data, print_full
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd


def main():
    housing_data = get_housing_data()
    head = housing_data.head()
    info = housing_data.info()

    pprint(info)  # prints the information with regards to the data frame
    pprint(head)  # prints the top 5 rows with regards to the data frame
    pprint(housing_data[
               "ocean_proximity"].value_counts())  # prints the count of all possible values `ocean_proximity` contains

    print_full(housing_data.describe())  # prints the statistical summary of the housing_data
    housing_data.hist(bins=50, figsize=(20, 15))

    # split the dataset between training sets and test sets.
    # use stratified approach when dealing with small sets of data since we could bias the sampling
    # ( we want the sets of data to represent the overall population)
    training_set, test_set = stratified_split(housing_data)
    training_set.hist(bins=50, figsize=(20, 15))
    test_set.hist(bins=50, figsize=(20, 15))
    plt.show()

    # visualizes data to try to see patterns
    visualize_housing_data(housing_data)
    plt.show()


def visualize_housing_data(data):
    cp = data.copy()
    cp.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=cp["population"] / 100, label="population",
            c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, figsize=(10, 7))


def stratified_split(data, test_size=0.2, seed=42):
    data["income_cat"] = pd.cut(data["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)
        return strat_train_set, strat_test_set


if __name__ == "__main__":
    main()
