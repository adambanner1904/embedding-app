from app.pipelines.embedding_models import get_bert_embeddings, get_tfidf_embeddings, get_doc2vec_embeddings
from app.pipelines.umap import get_short_embeddings


def doc2vec_pipeline(data, file_name):
    # Loads or creates/saves then returns long embeddings
    doc2vec_long_embeddings = get_doc2vec_embeddings(data, file_name)

    # Loads or creates/saves then returns short embeddings
    doc2vec_short_embeddings = get_short_embeddings(doc2vec_long_embeddings, file_name)

    # Returns x1, x2 to be plotted
    return doc2vec_short_embeddings[:, 0], doc2vec_short_embeddings[:, 1]


def preload_doc2vec_files(data):
    # Loads or creates/saves then returns long embeddings
    doc2vec_long_embeddings = get_doc2vec_embeddings(data)

    # Loads or creates/saves then returns short embeddings
    doc2vec_short_embeddings = get_short_embeddings(doc2vec_long_embeddings, 'doc2vec')


def bert_pipeline(data, file_name):
    # Loads or creates/saves then returns long embeddings
    bert_long_embeddings = get_bert_embeddings(data, file_name)

    # Loads or creates/saves then returns short embeddings
    bert_short_embeddings = get_short_embeddings(bert_long_embeddings, file_name)

    # Returns x1, x2 to be plotted
    return bert_short_embeddings[:, 0], bert_short_embeddings[:, 1]


def preload_bert_files(data):
    # Loads or creates/saves then returns long embeddings
    bert_long_embeddings = get_bert_embeddings(data)

    # Loads or creates/saves then returns short embeddings
    bert_short_embeddings = get_short_embeddings(bert_long_embeddings, 'bert')

    # Returns x1, x2 to be plotted
    return bert_short_embeddings[:, 0], bert_short_embeddings[:, 1]


def tfidf_pipeline(data, file_name):
    # Loads or creates/saves then returns long embeddings
    tfidf_long_embeddings = get_tfidf_embeddings(data, file_name)

    # Loads or creates/saves then returns short embeddings
    tfidf_short_embeddings = get_short_embeddings(tfidf_long_embeddings, file_name)

    # Returns x1, x2 to be plotted
    return tfidf_short_embeddings[:, 0], tfidf_short_embeddings[:, 1]


def preload_tfidf_files(data):
    # Loads or creates/saves then returns long embeddings
    tfidf_long_embeddings = get_tfidf_embeddings(data)

    # Loads or creates/saves then returns short embeddings
    tfidf_short_embeddings = get_short_embeddings(tfidf_long_embeddings, 'tfidf')

    # Returns x1, x2 to be plotted
    return tfidf_short_embeddings[:, 0], tfidf_short_embeddings[:, 1]
