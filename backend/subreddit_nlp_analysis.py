import numbers 
import re 
from collections import Counter
from pydantic import BaseModel
from typing import Dict, List, Tuple 
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import asyncio

# NLP related imports 
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy 
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Load the ABSA (Aspect Based Sentiment Analysis) model and tokenizer
model_name = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
summarizer= pipeline("summarization", model="facebook/bart-large-cnn")

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

async def get_subreddit_analysis(sorted_slice_to_posts): # called in main.py's FastAPI
    texts = []
    dates = []
    for date, posts in sorted_slice_to_posts.items():
        post_content = ' '.join([post.title + ' ' + post.description + ' '.join(post.comments) for post in posts])
        texts.append(post_content)
        dates.append(date)
    
    # Perform n-gram analysis - just only take ~ 1-2 seconds 
    t1 = time.time()
    top_n_grams = dict() 
    for i in range(0, len(dates)):
        filtered_post_content = preprocess_text_for_n_grams(texts[i]) # optimized and fast now 
        top_n_grams[dates[i]] = get_top_ngrams_sklearn([filtered_post_content]) # very fast 
    t2 = time.time()
    print('finished n-gram analysis in: ', t2-t1)

    # Perform NER analysis + sentiment analysis to EACH named entity 
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
    nlp.add_pipe('sentencizer')
    top_named_entities = dict()
    for date, doc in zip(dates, nlp.pipe(texts, batch_size=50)):
        top_named_entities[date] = doc

    tasks = [asyncio.create_task(postprocess_named_entities(date, doc)) for date, doc in top_named_entities.items()]
    results = await asyncio.gather(*tasks)

    for date, entities in results:
        top_named_entities[date] = entities

    t3 = time.time()
    print('finished NER analysis in: ', t3-t2)
    
    return top_n_grams, top_named_entities

async def postprocess_named_entities(date, doc):
    named_entities = Counter([ent.text for ent in doc.ents])
    banned_entities = {'one', 'two', 'first', 'second', 'yesterday', 'today', 'tomorrow', 
                       'approx', 'half', 'idk', 'congrats', 'three', 'creepy', 'night',
                       'day', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                       'saturday', 'sunday', 'month', 'months', 'day', 'days', 'week',
                       'weeks'}
    filtered_named_entities = Counter()
    for name, count in named_entities.items():
        if count < 3: continue # not enough sentences to sample 
        if not (isinstance(name, str) and filter_named_entity(name)): continue
        if isinstance(name, numbers.Number) or name.isnumeric() or name.lower().strip() in banned_entities: continue
        filtered_named_entities[name] = count
    
    top_ten_entities = filtered_named_entities.most_common(10)
    top_ten_entity_names = set([entity[0] for entity in top_ten_entities])
    # Write entity_to_sentences to a text file - I want to inspect the sentences individually
    #f = open("entity_to_sentences.txt", "w", encoding="utf-8")
    entity_to_sentiment, entity_to_sentences = await get_sentiments_of_entities(top_ten_entity_names, doc, f)
    #f.close()

    for i in range(len(top_ten_entities)):
        entity_name = top_ten_entities[i][0]
        entity_count = top_ten_entities[i][1]
        entity_sentiment = round(entity_to_sentiment[entity_name], 2)
        entity_sentences = entity_to_sentences[entity_name]
        top_ten_entities[i] = (entity_name, entity_count, entity_sentiment, entity_sentences) 

    return (date, top_ten_entities)

async def get_sentiments_of_entities(top_ten_entity_names, doc, f):
    entity_to_sentiment  = dict()
    entity_to_sentences = dict()
    entity_to_flattened_sentences = dict()
    entity_to_summary = dict() 
    for ent in set(doc.ents):
        if ent.text in top_ten_entity_names:
            if ent.text not in entity_to_sentences: entity_to_sentences[ent.text] = set()
            entity_to_sentences[ent.text].add(ent.sent.text)
    
    for entity, sentences in entity_to_sentences.items():
        flattened_sentences = "Comments regarding the entity \"" + entity + "\": \n"
        count = 1
        for sentence in sentences:
            flattened_sentences += str(count) + ".) " + sentence.strip() + "\n"
            count += 1
        #f.write(flattened_sentences + "\n")
        entity_to_flattened_sentences[entity] = flattened_sentences

    tasks = [asyncio.to_thread(get_sentiment_of_entity, entity, entity_to_flattened_sentences[entity]) for entity in entity_to_sentences]
    sentiment_results = await asyncio.gather(*tasks)
    for entity, sentiment_score in sentiment_results:
        entity_to_sentiment[entity] = sentiment_score
    
    tasks = [asyncio.to_thread(summarize_comments, entity, entity_to_flattened_sentences[entity]) for entity in entity_to_sentences]
    summarized_comments_results = await asyncio.gather(*tasks)
    for entity, summary in summarized_comments_results:
        entity_to_summary[entity] = summary

    return entity_to_sentiment, entity_to_summary

def get_sentiment_of_entity(entity, flattened_sentences):
    sentiment = classifier(flattened_sentences, text_pair=entity)
    # sentiment object example: [{'label': 'Positive', 'score': 0.9967294931411743}]
    # label = 'Positive', 'Neutral', or 'Negative' 
    # score = value in range [0, 1]
    sentiment_score = 0
    if(sentiment[0]['label'] == 'Positive'):
        sentiment_score = sentiment[0]['score']
    elif(sentiment[0]['label'] == 'Negative'):
        sentiment_score = (-1) * sentiment[0]['score']
    return (entity, sentiment_score)


def split_text(text, chunk_size=3000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def summarize_comments(entity, flattened_comments):
    #print('# characters in text: ', len(text))
    if len(flattened_comments) > 3000:
        # will probably exceed the token limit of 1024, so analyze each 3000 character chunk seperately
        chunks = split_text(flattened_comments)
        complete_summary = ""
        for chunk in chunks:
            summary = summarizer(chunk, max_length=100, min_length=20, do_sample=False)
            summary_text = summary[0]['summary_text']
            complete_summary += summary_text + "\n --- \n"
        return (entity, complete_summary)
    else:
        num_tokens = len(tokenizer.encode(flattened_comments))
        max_length = min(100, num_tokens)
        summary = summarizer(flattened_comments, max_length=max_length, min_length=5, do_sample=False)
        summary_text = summary[0]['summary_text']
        #print(summary_text)
        return (entity, summary_text)


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