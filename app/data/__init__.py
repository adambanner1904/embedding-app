import os
import numpy as np


def delete_all_saved_files():
    paths = ['app/static/long_embeddings/bert_embeddings.npy',
             'app/static/long_embeddings/doc2vec_embeddings.npy',
             'app/static/long_embeddings/tfidf_embeddings.npy',
             'app/static/short_embeddings/bert.npy',
             'app/static/short_embeddings/doc2vec.npy',
             'app/static/short_embeddings/tfidf.npy']
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def load_points(model, size):
    '''Model = [tfidf, doc2vec, bert]
        Size = [few=200, medium=2000, all=14949]'''
    path = f'app/static/short_embeddings/{size}_{model}.npy'
    points = np.load(path)
    return points[:, 0], points[:, 1]
