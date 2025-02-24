import numbers 
import re 
from collections import Counter
from pydantic import BaseModel
from typing import Dict, List, Tuple 

# NLP related imports 
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Perhaps naive filtering of named entities
# Returns true iff name is a valid named entity name
def filter_named_entity(name: str) -> bool:
    # Get parts and check if there is at least one part
    parts = name.split()
    if len(parts) == 0:
        return False

    # Check if each word is indeed a word, allowing for hyphenations and apostrophes
    # NOTE: May want to check for numbers as well
    # TODO: Allow ampersands in between "words"
    word_pattern = re.compile(r"^[a-zA-Z]+([-'’][a-zA-Z]+)*$")
    for part in parts:
        if not word_pattern.match(part):
            return False
    return True

# This function is called in the FastAPI and is what gets returned from FastAPI 
def get_subreddit_analysis(sorted_slice_to_posts):
    # Perform n-gram analysis 
    top_n_grams = dict() 
    for date, posts in sorted_slice_to_posts.items():
        post_content = ' '.join([post.title + ' '.join(post.comments) for post in posts])
        filtered_post_content = preprocess_text_for_n_grams(post_content)
        top_n_grams[date] = get_top_ngrams_sklearn([filtered_post_content])
    print('finished n-gram analysis')

    # Perform NER analysis 
    nlp = spacy.load("en_core_web_sm")
    top_named_entities = dict() 
    for date, posts in sorted_slice_to_posts.items():
        post_content = ' '.join([post.title + ' '.join(post.comments) for post in posts])
        doc = nlp(post_content)
        top_named_entities[date] = postprocess_named_entities(doc)
    print('finished NER analysis')
    
    return top_n_grams, top_named_entities

def postprocess_named_entities(doc):
    named_entities = Counter([ent.text for ent in doc.ents])
    # banned_entities = ['one', 'two', 'first', 'second', 'yesterday', 'today', 'tomorrow', 
    #                    'approx', 'half', 'idk']
    filtered_named_entities = Counter()
    for name, count in named_entities.items():
        if not (isinstance(name, str) and filter_named_entity(name)): continue
        # if isinstance(name, numbers.Number) or name.isnumeric() or name.lower().strip() in banned_entities: continue
        filtered_named_entities[name] = count
    
    filtered_top_named_entities = filtered_named_entities.most_common(10)
    # filtered_top_named_entities type = list of tuples with format (entity, # occurences)
    return filtered_top_named_entities 
    
def preprocess_text_for_n_grams(post_content):
    words = word_tokenize(post_content) # tokenize post_content 
    filtered_words = [re.sub(r'[^\w\s]', '', word) for word in words]  # Remove non-word characters
    filtered_words = [word for word in filtered_words if word]  # Remove empty strings
    filtered_words = [word.lower().strip() for word in words] # remove whitespace before and after word + make all lowercase 
    partial_stop_words = ['wiki', 'co', 'www.', 'reddit', '.com', 'autotldr', 'http']
    full_stop_words = ['or', 'and', 'you', 'that', 'the', 'co', 'en', 'np', 'removed', 'next']
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    filtered_words = [word for word in words if not any(exclude in word.lower() for exclude in partial_stop_words)]
    filtered_words = [word for word in filtered_words if word not in full_stop_words]
    filtered_post_content = ' '.join(filtered_words)
    return filtered_post_content 

# can parallelize this so for each (date, posts) pair runs concurrently 
def get_top_ngrams_sklearn(texts, n=3, top_k=10):
    vectorizer = CountVectorizer(ngram_range=(n, n))  # Extract only n-grams
    X = vectorizer.fit_transform(texts)  # Fit to the text
    freqs = X.sum(axis=0)  # Sum frequencies
    vocab = vectorizer.get_feature_names_out()
    counts = [(vocab[i], freqs[0, i]) for i in range(len(vocab))]
    return sorted(counts, key=lambda x: x[1], reverse=True)[:top_k]  # Sort and return top-k

def preprocess_text_for_sentiment(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# preprocess text first before calling get_sentiment
def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()

    scores = analyzer.polarity_scores(text)
    
    if scores["compound"] >= 0.05:
        return 1
    elif scores["compound"] <= -0.05:
        return -1
    else:
        return 0