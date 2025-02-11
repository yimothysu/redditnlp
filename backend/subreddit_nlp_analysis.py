import numbers 
import re 
from collections import Counter

# NLP related imports 
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy 

nltk.download('stopwords')
nltk.download('punkt')

class Subreddit_NLP_Analysis:
  def __init__(sorted_slice_to_posts, top_n_grams, top_named_entities):
     # returned data from fast api
     self.sorted_slice_to_posts = sorted_slice_to_posts 
     # dict where:
     #  key = date
     #  value = list of top n grams for slice using sklearn's CountVectorizer
     self.top_n_grams = top_n_grams 
     # dict where:
     # key = date  
     # value = list of top named entities for slice using spaCy 
     self.top_named_entities = top_named_entities 
    

# This function is called in the FastAPI and is what gets returned from FastAPI 
def get_subreddit_analysis(sorted_slice_to_posts):
    # response = requests.get(api_endpoint) # Call the FastAPI endpoint
    # data = response.json()

    # # Convert the JSON data back to the original format
    # restored_dict = {
    #     key: [RedditPost.model_validate(post) for post in posts]  # Convert each post to RedditPost
    #     for key, posts in data.items()
    # }

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
    
    # store everything in Subreddit_NLP_Analysis object and return it 
    subreddit_nlp_analysis = Subreddit_NLP_Analysis(
        sorted_slice_to_posts = sorted_slice_to_posts,
        top_n_grams = top_n_grams,
        top_named_entities = top_named_entities
    )
    return subreddit_nlp_analysis

def postprocess_named_entities(doc):
    named_entities = Counter([ent.text for ent in doc.ents])
    banned_entities = ['one', 'two', 'first', 'second', 'yesterday', 'today', 'tomorrow', 
                        'approx', 'half', 'idk']
    filtered_named_entities = Counter()
    for name, count in named_entities.items():
        if isinstance(name, numbers.Number) or name.isnumeric() or name.lower().strip() in banned_entities: continue
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