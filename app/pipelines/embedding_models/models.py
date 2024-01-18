from os import path

import gensim
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec
from sentence_transformers import SentenceTransformer


def tfidf():
    model = TfidfVectorizer(max_df=0.8, max_features=200000,
                            min_df=0.05, stop_words='english',
                            use_idf=True, ngram_range=(1, 3))
    return model


def doc2vec(vector_size=100):
    model = Doc2Vec(vector_size=vector_size, dm=0, window=2, workers=2, epochs=30)
    return model


def bert():
    if not path.exists('app/models/bert.pt'):
        model = SentenceTransformer("all-MiniLM-L12-v2")
        torch.save(model,'app/models/bert.pt')
    else:
        model = torch.load('app/models/bert.pt')
    return model


def get_tagged_docs(docs, tokens_only=False):
    for i, doc in enumerate(docs):
        tokens = gensim.utils.simple_preprocess(doc)
        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
