import pandas as pd
import plotly.express as px
import streamlit as st

from app.data import delete_all_saved_files, load_points
from app.data.data_tools import preprocess
from app.data.dataset_tools import get_dataset
from app.plotting import get_colour_key

if __name__ == '__main__':

    with st.sidebar:
        # Dataset changes
        st.subheader('Dataset')
        # Get info (st)
        size_choice = st.selectbox('Size of dataset:',
                                   options=['200', '2000', '14,949'])
        if size_choice == '200':
            size = 'few'
            number = 10
        elif size_choice == '2000':
            size = 'medium'
            number = 100
        elif size_choice == '14,949':
            size = 'all'
            number = None

        st.divider()

        # Model changes
        st.subheader('Model')
        chosen_model = st.selectbox('Model', options=['tfidf', 'doc2vec', 'bert'])

    dataset = get_dataset()
    # Apply reduction of dataset
    articles, categories = preprocess(dataset, 200, number)

    st.markdown(f'''Current size of the dataset: {len(articles)}''')

    x1, x2 = load_points(chosen_model, size)

    df = pd.DataFrame({'x': x1, 'y': x2, 'colour': categories})
    colour_key = get_colour_key()

    plot = px.scatter(df,
                      x='x',
                      y='y',
                      color='colour',
                      color_discrete_map=colour_key,
                      hover_name='colour',
                      opacity=0.7)

    # sentence = list(st.text_input('Write a sentence'))

    st.plotly_chart(plot)

    # if sentence:
    #     if chosen_model == 'tfidf':
    #         model = tfidf()
    #         x = model.fit_transform(sentence)
    #     if chosen_model == 'doc2vec':
    #         model = Doc2Vec.load('app/models/doc2vec')
    #         x = model.infer_vector(sentence).reshape((-1, 1))
    #     if chosen_model == 'bert':
    #         model = bert()
    #         x = model.encode(sentence)
    #     points = use_umap(x)
    #     x3, x4 = points[:, 0], points[:, 1]
    #     plot.add_scatter(x1, x2,
    #                      marker=dict(
    #                          color='red',
    #                          size=10,
    #                          symbol='x'
    #                      ))
