"""Module for generating word embeddings using Word2Vec and dimensionality reduction."""

import gensim.downloader as api # type: ignore
from sklearn.decomposition import PCA
import numpy as np

# Load pre-trained Word2Vec model
model = api.load("word2vec-google-news-300")


def get_word_embeddings(word_list):
    """Get word embeddings for a list of words using Word2Vec.
    
    Args:
        word_list: List of words to get embeddings for
    Returns:
        tuple: (list of words found in model, list of their embeddings)
    """
    word_embeddings = []
    words = []
    for word in word_list:
        if word in model.key_to_index:
            word_embeddings.append(model[word])
            words.append(word)
    return words, word_embeddings


def get_2d_embeddings(word_list):
    """Convert word embeddings to 2D coordinates using PCA.
    
    Args:
        word_list: List of words to get 2D embeddings for
    Returns:
        dict: Mapping of words to their 2D coordinates,
              or None if fewer than 2 words found in model
    """
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
