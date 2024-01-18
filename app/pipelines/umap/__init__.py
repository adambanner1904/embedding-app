from os import path

import numpy as np

from app.pipelines.umap.models import use_umap


def create_and_return_short_embeddings(data, name):

    short_embeddings = use_umap(data)
    np.save(f'app/static/short_embeddings/{name}', short_embeddings)
    return short_embeddings


def get_short_embeddings(articles, name):
    if path.exists(f'app/static/short_embeddings/{name}'):
        short_embeddings = np.load(f'app/static/short_embeddings/{name}')
    else:
        short_embeddings = create_and_return_short_embeddings(articles, name)

    return short_embeddings
