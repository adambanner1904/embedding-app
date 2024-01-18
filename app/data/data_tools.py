"""Functions that are going to take the entire dataset as an argument and return the data in a new format."""
import pandas as pd


def reduce_by_characters(dataset, characters=200):
    # articles, categories
    # print(dataset['articles'])

    mask = [len(t) > characters for t in dataset["articles"]]
    reduced_dataset = dataset[['articles', 'categories']][mask]
    return reduced_dataset


def reduce_to_n_samples(dataset, n_samples):
    if n_samples is None:
        return dataset.reset_index()
    else:
        reduced_dataset = dataset.groupby('categories', as_index=False).apply(lambda s: s.sample(n_samples))
        return reduced_dataset.reset_index()


def preprocess(data, characters_to_reduce_by, n_samples_to_reduce_to=None):
    # Apply reduction of dataset
    data = reduce_by_characters(data, characters_to_reduce_by)
    data = reduce_to_n_samples(data, n_samples_to_reduce_to)

    articles = data['articles']
    categories = data['categories']
    return articles, categories
