"""Tools to get the whole dataset from sklearn, store in local memory and read it from
local memory if it exists"""

from os import path

import numpy as np
import sklearn.datasets
import pandas as pd


def fetch_dataset():
    data = sklearn.datasets.fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    )

    articles = data['data']
    targets = data['target']
    categories = np.array([data.target_names[x] for x in targets])

    data = pd.DataFrame({'articles': articles, 'categories': categories})

    return data


fetch_dataset()


def load_dataset_locally():
    data = pd.read_csv('app/static/whole_dataset.csv')
    return data


def get_dataset():
    """If dataset does not exist locally then fetch it from sklearn and save it locally for later.
    Else load it from local storage and return it"""

    if not path.exists('app/static/whole_dataset.csv'):
        data = fetch_dataset()
        data.to_csv('app/static/whole_dataset.csv', index=False)

    else:
        data = load_dataset_locally()
    data = data.dropna()
    return data
