"""Module for extracting and analyzing n-grams from subreddit text content."""

import time, re, nltk # type: ignore
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
from src.utils.subreddit_classes import ( NGram )

nltk.download('stopwords')


def get_top_ngrams(post_content_grouped_by_date):
    """Extract top n-grams from posts grouped by date.
    
    Args:
        post_content_grouped_by_date: Dictionary mapping dates to post content
    Returns:
        dict: Mapping of dates to lists of NGram objects
    """
    dates = list(post_content_grouped_by_date.keys())
    post_content = list(post_content_grouped_by_date.values())
    
    t1 = time.time()
    top_n_grams = dict() 
    for i in range(0, len(dates)):
        filtered_post_content = preprocess_text_for_n_grams(post_content[i])
        top_bigrams = get_top_ngrams_sklearn([filtered_post_content])
        filtered_top_bigrams = []
        for top_bigram, count in top_bigrams:
            '''
                filters out bigrams that were mentioned less than 3 times +
                bigrams that have more than 2 consecutive digits (most likely a nonsense bigram --> Ex: 118 122)
            '''
            if count >= 3 and not bool(re.search(r'\d{3,}', top_bigram)):
                n_gram = NGram(
                    name = top_bigram,
                    count = count
                )
                filtered_top_bigrams.append(n_gram)
        top_n_grams[dates[i]] = filtered_top_bigrams 
    t2 = time.time()
    print('finished n-gram analysis in: ', t2-t1)
    return top_n_grams 


def preprocess_text_for_n_grams(post_content):
    """Clean and preprocess text for n-gram analysis.
    
    Removes stopwords, special characters, and common Reddit-specific terms.
    
    Args:
        post_content: Raw text content from posts
    Returns:
        str: Preprocessed text ready for n-gram extraction
    """
    partial_stop_words = {'wiki', 'co', 'www.', 'reddit', '.com', 'autotldr', 'http'}
    full_stop_words = {'or', 'and', 'you', 'that', 'the', 'co', 'en', 'np', 'removed', 'next'}
    full_stop_words = set(stopwords.words('english')).union(full_stop_words)
    
    words = word_tokenize(post_content)
    # Remove non-word characters, remove whitespace before and after word, make all lowercase
    filtered_words = [re.sub(r"[^\w\s']", '', word).lower().strip() for word in words] 
    filtered_words = [word for word in filtered_words if word]
    filtered_words = [word for word in filtered_words if word not in full_stop_words and 
                      not any(exclude in word for exclude in partial_stop_words)]
    filtered_post_content = ' '.join(filtered_words)
    return filtered_post_content 


def get_top_ngrams_sklearn(texts, n=2, top_k=10):
    """Extract top n-grams using scikit-learn's CountVectorizer.
    
    Args:
        texts: List of preprocessed text documents
        n: Size of n-grams to extract (default: 2 for bigrams)
        top_k: Number of top n-grams to return (default: 10)
    Returns:
        list: Top k n-grams with their frequencies, sorted by frequency
    """
    vectorizer = CountVectorizer(ngram_range=(n, n))  # Extract only n-grams
    X = vectorizer.fit_transform(texts)  # Fit to the text
    freqs = X.sum(axis=0)  # Sum frequencies
    vocab = vectorizer.get_feature_names_out()
    counts = [(vocab[i], freqs[0, i]) for i in range(len(vocab))]
    return sorted(counts, key=lambda x: x[1], reverse=True)[:top_k]  # Sort and return top-k
