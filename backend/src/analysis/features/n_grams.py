"""Module for extracting and analyzing n-grams from subreddit text content."""

import time, re, nltk, random, os # type: ignore
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
from src.utils.subreddit_classes import ( NGram )
from src.utils.subreddit_classes import ( RedditPost )

nltk.download('stopwords')

def load_banned_bigrams():
    project_root = os.path.abspath(os.path.join(__file__, "../../../../"))
    file_path = os.path.join(project_root, "data", "banned_bigrams.txt")
    banned_bigrams = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            banned_bigrams.add(line.strip().lower())
    return banned_bigrams

def get_flattened_text_for_post(post: RedditPost, include_comments=False):
    flattened_text = "Title: {title}\nDescription: {description}".format(title = post.title, description = post.description)
    if include_comments: 
        flattened_text += "Comments: \n"
        for top_level_comment in random.sample(post.top_level_comments, min(100, len(post.top_level_comments))):
            flattened_text += " " + top_level_comment.text
    return flattened_text 


def get_top_ngrams(posts: list[RedditPost]):
    """Extract top n-grams from posts 
    """
    banned_bigrams = load_banned_bigrams() 
    ngram_to_freq = dict() # key = n-gram, value = frequency of n-gram 
    for i in range(0, len(posts)):
        post_content = get_flattened_text_for_post(posts[i], include_comments=True) 
        filtered_post_content = preprocess_text_for_n_grams(post_content)
        top_ngrams_for_post = get_top_ngrams_sklearn([filtered_post_content])
        for top_ngram, count in top_ngrams_for_post:
            '''
                filters out bigrams that were mentioned less than 3 times +
                bigrams that have more than 2 consecutive digits (most likely a nonsense bigram --> Ex: 118 122)
            '''
            if count >= 3 and not bool(re.search(r'\d{3,}', top_ngram)) and top_ngram not in banned_bigrams:
                if top_ngram not in ngram_to_freq:
                    ngram_to_freq[top_ngram] = 0
                ngram_to_freq[top_ngram] += count 
    # convert ngram_to_freq to a list of NGram objects 
    top_ngrams = []
    for ngram, freq in ngram_to_freq.items():
        top_ngrams.append(NGram(name=ngram, count=freq))
    # Sort top_ngrams by freq and pick the top 50 n-grams 
    sorted_top_ngrams = sorted(top_ngrams, key=lambda x: x.count)
    sorted_top_ngrams = sorted_top_ngrams[-min(50, len(top_ngrams)):]
    return sorted_top_ngrams 


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
