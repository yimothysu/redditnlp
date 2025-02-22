import numbers 
import re 
from collections import Counter
from pydantic import BaseModel
from typing import Dict, List, Tuple 
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time

# NLP related imports 
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy 

start = time.time()
nltk.download('stopwords')
nltk.download('punkt')
end = time.time()
print('finished downloading nltk stopwords and punkt in: ', end-start)

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

def get_subreddit_analysis(sorted_slice_to_posts): # called in main.py's FastAPI
    texts = []
    dates = []
    for date, posts in sorted_slice_to_posts.items():
        post_content = ' '.join([post.title + ' '.join(post.comments) for post in posts])
        texts.append(post_content)
        dates.append(date)
    # Perform n-gram analysis - just only take ~ 1-2 seconds 
    # ProcessPoolExecutor and ThreadPoolExecuter do NOT help speed up n-gram analysis 
    t1 = time.time()
    top_n_grams = dict() 
    for i in range(0, len(dates)):
        filtered_post_content = preprocess_text_for_n_grams(texts[i]) # optimized and fast now 
        top_n_grams[dates[i]] = get_top_ngrams_sklearn([filtered_post_content]) # very fast 
    t2 = time.time()
    print('finished n-gram analysis in: ', t2-t1)

    # Perform NER analysis - takes ~ 20 seconds 
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
    top_named_entities = dict() 
    for doc, date in zip(nlp.pipe(texts, batch_size=50), dates): # Processing all texts in batches
        top_named_entities[date] = postprocess_named_entities(doc)
    t3 = time.time()
    print('finished NER analysis in: ', t3-t2)
    
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
    partial_stop_words = {'wiki', 'co', 'www.', 'reddit', '.com', 'autotldr', 'http'}
    full_stop_words = {'or', 'and', 'you', 'that', 'the', 'co', 'en', 'np', 'removed', 'next'}
    full_stop_words = set(stopwords.words('english')).union(full_stop_words)
    
    words = word_tokenize(post_content) # tokenize post_content 
    # Remove non-word characters, remove whitespace before and after word, make all lowercase
    filtered_words = [re.sub(r'[^\w\s]', '', word).lower().strip() for word in words] 
    filtered_words = [word for word in filtered_words if word]  # Remove empty strings
    filtered_words = [word for word in filtered_words if word not in full_stop_words and 
                      not any(exclude in word for exclude in partial_stop_words)]
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