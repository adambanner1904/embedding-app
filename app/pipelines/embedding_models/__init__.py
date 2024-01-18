from os import path

import numpy as np

from app.pipelines.embedding_models.models import tfidf, doc2vec, bert, get_tagged_docs


def create_and_return_tfidf_embeddings(articles, file_name, create=True):
    model = tfidf()
    x = model.fit_transform(articles).toarray()

    # Save them
    if create:
        np.save(f'app/static/long_embeddings/{file_name}.npy', x)
    return x


def get_tfidf_embeddings(articles, file_name):
    if path.exists(f'app/static/long_embeddings/{file_name}.npy'):
        tfidf_long_embeddings = np.load(f'app/static/long_embeddings/{file_name}.npy')
    else:
        tfidf_long_embeddings = create_and_return_tfidf_embeddings(articles, file_name)

    return tfidf_long_embeddings


def create_and_return_doc2vec_embeddings(articles, file_name, create=True):
    model = doc2vec()
    tagged_docs = list(get_tagged_docs(articles))
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    x = np.array([model.dv[i] for i in range(len(model.dv))])
    if create:
        np.save(f'app/static/long_embeddings/{file_name}.npy', x)
    return x


def get_doc2vec_embeddings(articles, file_name):
    if path.exists(f'app/static/long_embeddings/{file_name}.npy'):
        doc2vec_long_embeddings = np.load(f'app/static/long_embeddings/{file_name}.npy')
    else:
        doc2vec_long_embeddings = create_and_return_doc2vec_embeddings(articles, file_name)
    return doc2vec_long_embeddings


def create_and_return_bert_embeddings(articles, file_name, create=True):
    model = bert()
    x = model.encode(articles, show_progress_bar=False)
    if create:
        np.save(f'app/static/long_embeddings/{file_name}.npy', x)
    return x


def get_bert_embeddings(articles, file_name):
    if path.exists(f'app/static/long_embeddings/{file_name}.npy'):
        bert_long_embeddings = np.load(f'app/static/long_embeddings/{file_name}.npy')
    else:
        bert_long_embeddings = create_and_return_bert_embeddings(articles, file_name)

    return bert_long_embeddings
