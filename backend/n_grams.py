from sklearn.feature_extraction.text import CountVectorizer
import requests
import re 
from main import RedditPost  

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def get_top_ngrams_sklearn(texts, n=3, top_k=10):
    vectorizer = CountVectorizer(ngram_range=(n, n))  # Extract only n-grams
    X = vectorizer.fit_transform(texts)  # Fit to the text
    freqs = X.sum(axis=0)  # Sum frequencies
    vocab = vectorizer.get_feature_names_out()
    counts = [(vocab[i], freqs[0, i]) for i in range(len(vocab))]
    return sorted(counts, key=lambda x: x[1], reverse=True)[:top_k]  # Sort and return top-k

# Example usage
texts = ["This is a test. This is only a test. This test is simple."]
print('hi')
print(get_top_ngrams_sklearn(texts, n=2, top_k=10))

# Define the API endpoint (adjust if needed)
url = "http://localhost:8000/sample/technology"

# Call the FastAPI endpoint
response = requests.get(url)
data = response.json()
# Convert the JSON data back to the original format
restored_dict = {
    key: [RedditPost.model_validate(post) for post in posts]  # Convert each post to RedditPost
    for key, posts in data.items()
}

for date, posts in restored_dict.items():
    post_content = ' '.join([post.title + ' '.join(post.comments) for post in posts])
    
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
    
    print('key: ', date, ', top_ngrams: ', get_top_ngrams_sklearn([filtered_post_content]))
