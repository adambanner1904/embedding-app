import numpy as np

from app.data import delete_all_saved_files
from app.data.data_tools import preprocess, reduce_by_characters
from app.data.dataset_tools import get_dataset
from app.pipelines import tfidf_pipeline, doc2vec_pipeline, bert_pipeline
from app.pipelines.embedding_models import get_bert_embeddings, get_tfidf_embeddings, get_doc2vec_embeddings, doc2vec, \
    get_tagged_docs, tfidf
from app.pipelines.umap import get_short_embeddings

'''This file was for saving all the short embeddings so I can just load them instead of recreate them all the time. 
The user only really needs a few options, one in the hundreds, in the thousands and then tens of thousans.
Small = 200
Medium = 2000
All = 14949 after reduction of articles with less than 200 characters'''
dataset = get_dataset()
#
# few_articles, few_categories = preprocess(dataset, 200, 10)

# medium_articles, medium_categories = preprocess(dataset, 200, 100)
#
all_articles, all_categories = preprocess(dataset, 200)

# x1, x2 = tfidf_pipeline(all_articles, 'all_tfidf')
# x3, x4 = doc2vec_pipeline(all_articles, 'all_doc2vec')
# x5, x6 = bert_pipeline(all_articles, 'all_bert')

# Save doc2vec model for inferring vectors
# model = doc2vec()
# tagged_docs = list(get_tagged_docs(all_articles))
# model.build_vocab(tagged_docs)
# model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
# model.save('app/models/doc2vec')


# Pickling tfidf model
# model = tfidf()
# model.fit_transform(all_articles)
# print(dataset.head())

bert = np.load('app/static/long_embeddings/few_bert.npy')
doc2vec = np.load('app/static/long_embeddings/few_doc2vec.npy')
tfidf = np.load('app/static/long_embeddings/few_tfidf.npy')

print(f'bert shape: {bert.shape}')
print(f'doc2vec shape: {doc2vec.shape}')
print(f'tfidf shape: {tfidf.shape}')
