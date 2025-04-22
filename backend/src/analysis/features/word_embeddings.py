import gensim.downloader as api # type: ignore
from sklearn.decomposition import PCA
import numpy as np

model = api.load("word2vec-google-news-300")

def get_word_embeddings(word_list):
    word_embeddings = []
    words = []
    for word in word_list:
        if word in model.key_to_index:
            word_embeddings.append(model[word])
            words.append(word)
    return words, word_embeddings

def get_2d_embeddings(word_list):
    words, embeddings = get_word_embeddings(word_list)
    if len(words) <= 1:
        return None
    if embeddings:
        pca = PCA(n_components=2)
        np_embedding = np.array(embeddings)
        coordinates_2d = pca.fit_transform(np_embedding)

        word_to_coordinates = {}
        for word, embedding in zip(words, coordinates_2d):
            word_to_coordinates[word] = (embedding[0], embedding[1])
        return word_to_coordinates
    return None