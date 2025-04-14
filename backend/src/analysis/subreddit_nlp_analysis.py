import numbers 
import re 
from collections import Counter
import time
import asyncio
import re
import os 
import math 
import random 
import asyncpraw
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from readability import Readability
from collections import defaultdict

# NLP related imports 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy 
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from src.data_fetchers.reddit_post_fetcher import (
    fetch_post_data,
)

from src.utils.subreddit_classes import (
    SubredditQuery,
    SubredditAnalysis,
)

from src.analysis.word_embeddings import (
    get_2d_embeddings,
)

from src.analysis.word_cloud import (
    generate_word_cloud,
)

# Topic extraction with BERTopic
from bertopic import BERTopic

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Load the ABSA (Aspect Based Sentiment Analysis) model and tokenizer
absa_model_name = "yangheng/deberta-v3-base-absa-v1.1"
absa_model = AutoModelForSequenceClassification.from_pretrained(absa_model_name)
absa_tokenizer = AutoTokenizer.from_pretrained(absa_model_name)
classifier = pipeline("text-classification", model=absa_model, tokenizer=absa_tokenizer)
# Load the Summarizer model and tokenizer 
summarizer= pipeline("summarization", model="facebook/bart-large-cnn")
summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# TIME_FILTER_TO_POST_LIMIT = {'all': 400, 'year': 200, 'month': 100, 'week': 50}
# Strategy: First query a bunch of posts up to the limit defined by TIME_FILTER_TO_INITIAL_POST_QUERY_LIMIT. 
# Then throw away the posts which have an empty description (means OP posted a link or a photo.) 
# Then with the remaining posts, query the comments of posts up to the limit defined by TIME_FILTER_TO_POST_LIMIT 
TIME_FILTER_TO_INITIAL_POST_QUERY_LIMIT = {'all': 1000, 'year': 1000, 'week': 200}
TIME_FILTER_TO_POST_LIMIT = {'all': 350, 'year': 200, 'week': 50}
TIME_FILTER_TO_NAMED_ENTITIES_LIMIT = {'all': 12, 'year': 10, 'week': 8}

load_dotenv()

config = {
    "time_filter": "week"
}

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

# Get embedding for named entity by averaging embeddings over all words
def get_entity_embedding(entity):
    words = entity.split()
    vectors = [absa_model[word] for word in words if word in absa_model]
    if vectors:
        return np.mean(vectors, axis=0)
    return None

def cluster_similar_entities(named_entities):
    entity_embeddings = []
    entity_names = []
    for entity in named_entities:
        embedding = get_entity_embedding(entity)
        # Question: Do we only want to consider entities that have an embedding?
        if embedding is not None:
            entity_embeddings.append(embedding)
            entity_names.append(entity)

    # If we only have one entity embedding then it's not enough to cluster
    if len(entity_embeddings) < 2:
        return named_entities

    entity_embeddings = np.array(entity_embeddings)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.4, metric='cosine', linkage='average')
    labels = clustering.fit_predict(entity_embeddings)
    clustered_entities = {}
    for i, label in enumerate(labels):
        clustered_entities.setdefault(label, []).append(entity_names[i])

    # We choose the most frequnent entity as the representative for the cluster
    entity_to_cluster = {}
    for entities in clustered_entities.values():
        most_common_entity = max(entities, key=lambda e: named_entities[e])
        entity_to_cluster[most_common_entity] = named_entities[most_common_entity]
    return entity_to_cluster

def group_named_entities():
    print('meow')

# gets the top post urls for each named entity so users can reference the posts 
def get_post_urls(date_to_posts, date_to_entities):
    t1 = time.time()
    time_filter_to_post_url_limit = {'week': 1, 'year': 3, 'all': 3}
    num_urls_max = time_filter_to_post_url_limit[config['time_filter']]
    new_date_to_entities = {}
    for date, entities in date_to_entities.items():
        for entity in entities:
            entity_name = entity[0]
            posts_that_mention_entity = [] # idxs of the posts 
            for post in date_to_posts[date]:
                post_content = ' '.join([post.title + ' ' + post.description + ' '.join(post.comments)])
                if entity_name in post_content: 
                    posts_that_mention_entity.append((post.url, post_content.count(entity_name), post.score))
            # Time to pick which posts to feature 
            # sort posts_that_mention_entity by post_content.count(entity_name) * score 
            posts_that_mention_entity.sort(key=lambda post: post[1] * post[2])
            post_urls_for_entity = [post[0] for post in posts_that_mention_entity[-num_urls_max:]]
            if date not in new_date_to_entities:
                new_date_to_entities[date] = []
            new_date_to_entities[date].append((entity + (post_urls_for_entity,)))
    t2 = time.time()
    print('get_post_urls took: ', t2 - t1)
    return new_date_to_entities


async def get_subreddit_analysis(sorted_slice_to_posts, set_progress): # called in main.py's FastAPI
    texts = []
    dates = []
    for date, posts in sorted_slice_to_posts.items():
        post_content = ' '.join([post.title + ' ' + post.description + ' '.join(post.comments) for post in posts])
        if len(post_content) >= 1000000:
            # randomly sample the comments 
            comments = []
            for post in posts:
                half_of_comments = random.sample(post.comments, k=len(comments) // 2)
                comments.extend(half_of_comments)
            post_content = ' '.join([post.title + ' ' + post.description + ' '.join(comments)])
        texts.append(post_content)
        dates.append(date)
    
    # Perform n-gram analysis 
    t1 = time.time()
    top_n_grams = dict() 
    for i in range(0, len(dates)):
        filtered_post_content = preprocess_text_for_n_grams(texts[i]) # optimized and fast now 
        top_bigrams = get_top_ngrams_sklearn([filtered_post_content]) # very fast 
        filtered_top_bigrams = []
        for top_bigram, count in top_bigrams:
            # filters out bigrams that were mentioned less than 3 times +
            # bigrams that have more than 2 consecutive digits (most likely a nonsense 
            # bigram --> Ex: 118 122)
            if count >= 3 and not bool(re.search(r'\d{3,}', top_bigram)):
                filtered_top_bigrams.append((top_bigram, count))
        top_n_grams[dates[i]] = filtered_top_bigrams 
    t2 = time.time()
    print('finished n-gram analysis in: ', t2-t1)

    # Perform NER analysis + sentiment analysis to EACH named entity + get top post urls for each named entity 
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
    nlp.add_pipe('sentencizer')
    t3 = time.time()
    print('initializing nlp took: ', t3 - t2)

    top_named_entities = dict()
    for date, doc in zip(dates, nlp.pipe(texts, batch_size=50)):
        top_named_entities[date] = doc
    t4 = time.time()
    print('getting top named entities using nlp took: ', t4 - t3)

    progress_list = [0] * len(top_named_entities.items())
    def set_item_progress(idx, progress):
        nonlocal progress_list
        progress_list[idx] = progress
        set_progress(sum(progress_list) / len(top_named_entities.items()))
    
    tasks = [
        asyncio.create_task(postprocess_named_entities(date, doc, lambda progress: set_item_progress(idx, progress)))
        for idx, (date, doc) in enumerate(top_named_entities.items())
    ]

    results = await asyncio.gather(*tasks)

    for date, entities in results:
        top_named_entities[date] = entities

    top_named_entities = get_post_urls(sorted_slice_to_posts, top_named_entities)

    t5 = time.time()
    print('finished NER analysis in: ', t5-t2)
    return top_n_grams, top_named_entities


async def postprocess_named_entities(date, doc, set_progress):
    t1 = time.time()
    named_entities = Counter([ent.text for ent in doc.ents])
    banned_entities = {'one', 'two', 'first', 'second', 'yesterday', 'today', 'tomorrow', 
                       'approx', 'half', 'idk', 'congrats', 'three', 'creepy', 'night',
                       'day', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                       'saturday', 'sunday', 'month', 'months', 'day', 'days', 'week',
                       'weeks', 'this year', 'next year', 'last year', 'year', 'a year', 
                       'years', 'a couple years', 'the week', 'the summer', 'spring', 
                       'summer', 'fall', 'winter', 'zero', 'one day', 'the next day', 
                       'six', 'seven', 'eight', 'the years', 'every day', "n't", 'that day',
                       'one year', 'each day', 'last week', 'all day', 'a couple of weeks', 'daily',
                       'some days', 'hours', 'a week', 'some days', 'the day', 'a few years', 'every morning',
                       'morning', 'an hour', 'a month', 'a year ago', 'most days', 'tonight', 'overnight', 
                       'the end of the day', 'kinda', 'last month', 'a few minutes', 'dude', 'those years',
                       'n’t', 'lot', 'only two', 'ten', 'the morning', 'five years', 'op', 'a ton', 'a few years ago',
                       'decades', 'a few months', 'a few days', 'over a decade', 'this summer', 'a bad day',
                       'the days', 'these days', 'shit', 'fuck', 'ass', 'asshole', 'bitch',
                       'three years', 'two years', 'one year', 'every single day', 'this day',
                       'thousands', 'af', 'weekly', 'a decade', 'another day', 'each year', 'each week',
                       'ten years', 'january', 'february', 'march', 'april', 'may', 'june',
                       'july', 'august', 'september', 'october', 'november', 'december', 'four',
                       'five', 'nine', 'this week', 'next week', 'a few more years', 'a new day', 
                       'the other day', 'third', 'every night', 'max', 'nah', 'yeah', 'yea', 'rip', 
                       'last night', 'six months', 'years later', 'about a week', 'more than one', 'depends', 
                       'about a year', 'every night', 'monthly', 'each month', 'one night', 'the night', 'the year', 
                       'a few weeks', 'nights', 'that night', 'a few hours', 'that year', 'the first year', 'time',
                       'a couple of years'}
    # clustered_named_entites = cluster_similar_entities(named_entities)
    filtered_named_entities = Counter()
    for name, count in named_entities.items():
        if count < 3: continue # not enough sentences to sample 
        if not (isinstance(name, str) and filter_named_entity(name)): continue
        if isinstance(name, numbers.Number) or name.isnumeric() or name.lower().strip() in banned_entities: continue
        filtered_named_entities[name] = count
    t2 = time.time()


    top_ten_entities = filtered_named_entities.most_common(TIME_FILTER_TO_NAMED_ENTITIES_LIMIT[config['time_filter']])
    top_ten_entity_names = set([entity[0] for entity in top_ten_entities])
    # Write entity_to_sentences to a text file - I want to inspect the sentences individually
    #f = open("entity_to_sentences.txt", "w", encoding="utf-8")
    entity_to_sentiment, entity_to_sentences = await get_sentiments_of_entities(top_ten_entity_names, doc)
    t3 = time.time()

    print('getting sentiment + summary of named entities in ', date, ' took ', t3 - t2)
    #f.close()

    for i in range(len(top_ten_entities)):
        entity_name = top_ten_entities[i][0]
        entity_count = top_ten_entities[i][1]
        if entity_name in entity_to_sentiment:
            entity_sentiment = round(entity_to_sentiment[entity_name], 2)
            entity_sentences = entity_to_sentences[entity_name]
            top_ten_entities[i] = (entity_name, entity_count, entity_sentiment, entity_sentences)
        else:
            top_ten_entities[i] = None
    
    top_ten_entities = [top_entity for top_entity in top_ten_entities if top_entity is not None]

    set_progress(1.0)

    return (date, top_ten_entities)

async def get_sentiments_of_entities(top_ten_entity_names, doc):
    entity_to_sentiment  = dict()
    entity_to_sentences = dict()
    entity_to_flattened_sentences = dict()
    entity_to_summary = dict() 
    for ent in doc.ents:
        ent_text = ent.text 
        if ent_text in top_ten_entity_names:
            sent_text = ent.sent.text 
            if ent_text not in entity_to_sentences: 
                entity_to_sentences[ent_text] = set()
            entity_to_sentences[ent_text].add(sent_text)
    
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
    sentiment_results = [sentiment_result for sentiment_result in sentiment_results if sentiment_result is not None]
    for entity, sentiment_score in sentiment_results:
        entity_to_sentiment[entity] = sentiment_score
    
    tasks = [asyncio.to_thread(summarize_comments, entity, entity_to_flattened_sentences[entity]) for entity in entity_to_sentences]
    summarized_comments_results = await asyncio.gather(*tasks)
    summarized_comments_results = [result for result in summarized_comments_results if result is not None]
    for entity, summary in summarized_comments_results:
        entity_to_summary[entity] = summary
    
    entity_to_sentiment, entity_to_summary = combine_same_entities(entity_to_sentiment, entity_to_summary)
    return entity_to_sentiment, entity_to_summary


def combine_entity_summaries_and_sentiments(entity_to_sentiment, entity_to_summary, entity_1, entity_2):
    combined_sentiment = (entity_to_sentiment[entity_1] + entity_to_sentiment[entity_2]) / 2
    entity_to_sentiment[entity_1] = combined_sentiment
    del entity_to_sentiment[entity_2]

    combined_summary = entity_to_summary[entity_1] + " \n " + entity_to_summary[entity_2]
    _, combined_summary = summarize_comments(entity_1, combined_summary)
    entity_to_summary[entity_1] = combined_summary
    del entity_to_summary[entity_2]

    return entity_to_sentiment, entity_to_summary


def combine_same_entities(entity_to_sentiment, entity_to_summary):
    # Combine entities that refer to the same thing
    # Ex: (republican, republicans), (Democrat, democrats), (US, USA, America), (American, Americans) 
    # Any pair of entities that are the same if you lowercase the first letter —> Ex: China, china
    
    print('inside combine_same_entities')
    for entity in list(entity_to_summary.keys()):
        entity_lowercase = entity[0].lower() + entity[1:]
        entity_lowercase_plural = entity[0].lower() + entity[1:] + "s"
        entity_plural = entity + "s"
        
        if entity_lowercase in entity_to_summary and entity != entity_lowercase:
            print('combining ', entity, ' and ', entity_lowercase)
            entity_to_sentiment, entity_to_summary = combine_entity_summaries_and_sentiments(entity_to_sentiment, entity_to_summary, entity, entity_lowercase)
        
        if entity_lowercase_plural in entity_to_summary:
            print('combining ', entity, ' and ', entity_lowercase_plural)
            entity_to_sentiment, entity_to_summary = combine_entity_summaries_and_sentiments(entity_to_sentiment, entity_to_summary, entity, entity_lowercase_plural)
        
        if entity_plural in entity_to_summary:
            print('combining ', entity, ' and ', entity_plural)
            entity_to_sentiment, entity_to_summary = combine_entity_summaries_and_sentiments(entity_to_sentiment, entity_to_summary, entity, entity_plural)

        america_synonyms = ["USA", "US", "America", "america", "usa"]
        if entity in america_synonyms and entity in entity_to_summary:
            america_synonyms.remove(entity)
            sentiments = [entity_to_sentiment[entity]]
            summaries = [entity_to_summary[entity]]
            
            for america_synonym in america_synonyms:
                if america_synonym in entity_to_summary:
                    sentiments.append(entity_to_sentiment[america_synonym])
                    summaries.append(entity_to_summary[america_synonym])
                    del entity_to_sentiment[america_synonym]
                    del entity_to_summary[america_synonym]
            
            if len(sentiments) > 1:
                print('combining ', entity, ' with ', len(sentiments), ' synonyms')
                combined_sentiment = sum(sentiments) / len(sentiments)
                _, combined_summary = summarize_comments(entity, ' \n '.join(summaries))
                entity_to_sentiment[entity] = combined_sentiment
                entity_to_summary[entity] = combined_summary

    return entity_to_sentiment, entity_to_summary 
    

def split_string_into_chunks(text, num_chunks):
    words = text.split()
    avg = math.ceil(len(words) / num_chunks)
    chunks = [" ".join(words[i * avg:(i + 1) * avg]) for i in range(num_chunks)]
    return [chunk for chunk in chunks if chunk]

def get_sentiment_of_entity(entity, flattened_sentences):
    absa_tokenizer = AutoTokenizer.from_pretrained(absa_model_name)
    tokens = absa_tokenizer(flattened_sentences, truncation=False)
    if(len(tokens['input_ids']) >= 2000):
        t1 = time.time()
        num_chunks = math.ceil(len(tokens['input_ids']) / 2000)
        chunks = split_string_into_chunks(flattened_sentences, num_chunks)
        averaged_sentiment_score = 0
        
        for chunk in chunks:
            tokens = absa_tokenizer(chunk, truncation=False)
            print("Number of tokens:", len(tokens['input_ids']))
            try:
                sentiment = classifier(chunk, text_pair=entity)
            except:
                print('couldnt get the sentiment score of ', entity) 
                return None 

            sentiment_score = 0
            if(sentiment[0]['label'] == 'Positive'):
                sentiment_score = sentiment[0]['score']
            elif(sentiment[0]['label'] == 'Negative'):
                sentiment_score = (-1) * sentiment[0]['score']
            averaged_sentiment_score += sentiment_score 
        
        averaged_sentiment_score = averaged_sentiment_score / num_chunks 
        t2 = time.time() 
        print('getting sentiment of the entity ', entity, ' with ', num_chunks, ' chunks took ', t2 - t1)
    else:
        tokens = absa_tokenizer(flattened_sentences, truncation=False)
        print("Number of tokens:", len(tokens['input_ids']))
        t1 = time.time()
        try:
            sentiment = classifier(flattened_sentences, text_pair=entity)
        except:
            print('couldnt get the sentiment score of ', entity) 
            return None 
        t2 = time.time()
    print('getting sentiment of the entity ', entity, ' with 1 chunk took ', t2 - t1)
    # sentiment object example: [{'label': 'Positive', 'score': 0.9967294931411743}]
    # label = 'Positive', 'Neutral', or 'Negative' 
    # score = value in range [0, 1]
    sentiment_score = 0
    if(sentiment[0]['label'] == 'Positive'):
        sentiment_score = sentiment[0]['score']
    elif(sentiment[0]['label'] == 'Negative'):
        sentiment_score = (-1) * sentiment[0]['score']
    return (entity, sentiment_score)


def split_text(text, chunk_size=2500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def summarize_comments(entity, flattened_comments):
    #print('# characters in text: ', len(text))
    summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    time_filter_to_key_points_limit = {'week': 3, 'month': 3, 'year': 5, 'all': 7}
    key_points_limit = time_filter_to_key_points_limit[config['time_filter']]
    complete_summary = ""
    if len(flattened_comments) > 2500:
        t1 = time.time()
        # will probably exceed the token limit of 1024, so analyze each 3000 character chunk seperately
        chunks = split_text(flattened_comments)
        sentences_summary = []
        for chunk in chunks:
            tokenized = summarizer_tokenizer.encode(chunk, truncation=True, max_length=1024)
            truncated_chunk = summarizer_tokenizer.decode(tokenized)
            print('truncated_chunk # tokens: ', len(summarizer_tokenizer.encode(truncated_chunk)))
            summary = summarizer(truncated_chunk, max_length=100, min_length=20, do_sample=False)
            summary_text = summary[0]['summary_text']
            
            sentences = nltk.sent_tokenize(summary_text)
            # Only keep the sentences in summary_text which mention the entity 
            relevant_sentences = [sentence for sentence in sentences if entity in sentence]
            sentences_summary.extend(relevant_sentences)
        
        if len(sentences_summary) > key_points_limit and len(sentences_summary) <= key_points_limit + 5:
            key_points = random.sample(sentences_summary, key_points_limit)
            complete_summary = "\n --- \n".join(key_points)
        elif len(sentences_summary) > key_points_limit + 5: # happens for an entity that is mentioned in A LOT of comments 
            print("computing a summary of summary")
            _, summary_of_summary = summarize_comments(entity, "\n".join(sentences_summary))
            complete_summary = summary_of_summary
        else:
            complete_summary = "\n --- \n".join(sentences_summary)

        t2 = time.time()
        print('getting summarized sentiment of entity (len(flattened_comments) > 3000)', entity, ' took ', t2 - t1)
        return (entity, complete_summary)
    else:
        t1 = time.time()
        tokenized = summarizer_tokenizer.encode(flattened_comments, truncation=True, max_length=1024)
        truncated_flattened_comments = summarizer_tokenizer.decode(tokenized) 
        print('truncated_flattened_comments # tokens: ', len(summarizer_tokenizer.encode(truncated_flattened_comments)))
        num_tokens = len(tokenized)
        max_length = min(100, num_tokens)
        summary = summarizer(truncated_flattened_comments, max_length=max_length, min_length=5, do_sample=False)
        summary_text = summary[0]['summary_text']

        sentences = nltk.sent_tokenize(summary_text)
        # Only keep the sentences in summary_text which mention the entity 
        sentences_summary = [sentence for sentence in sentences if entity in sentence]
        if len(sentences_summary) > key_points_limit:
            key_points = random.sample(sentences_summary, key_points_limit)
            complete_summary = "\n --- \n".join(key_points)
        else:
            complete_summary = "\n --- \n".join(sentences_summary)
        t2 = time.time()
        print('getting summarized sentiment of entity ', entity, ' took ', t2 - t1)
        return (entity, complete_summary)


def preprocess_text_for_n_grams(post_content):
    partial_stop_words = {'wiki', 'co', 'www.', 'reddit', '.com', 'autotldr', 'http'}
    full_stop_words = {'or', 'and', 'you', 'that', 'the', 'co', 'en', 'np', 'removed', 'next'}
    full_stop_words = set(stopwords.words('english')).union(full_stop_words)
    
    words = word_tokenize(post_content) # tokenize post_content 
    # Remove non-word characters, remove whitespace before and after word, make all lowercase
    filtered_words = [re.sub(r"[^\w\s']", '', word).lower().strip() for word in words] 
    filtered_words = [word for word in filtered_words if word]  # Remove empty strings
    filtered_words = [word for word in filtered_words if word not in full_stop_words and 
                      not any(exclude in word for exclude in partial_stop_words)]
    filtered_post_content = ' '.join(filtered_words)
    return filtered_post_content 

# can parallelize this so for each (date, posts) pair runs concurrently 
def get_top_ngrams_sklearn(texts, n=2, top_k=10):
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

# Topic extraction with BERTopic
def extract_topics(texts, num_topics=10):
    """
    Extract topics from a list of texts using BERTopic.
    
    Args:
        texts (list): List of text documents for topic modeling
        num_topics (int, optional): Number of topics to extract. Defaults to 10.
        
    Returns:
        dict: A dictionary containing:
            - topics: List of topic numbers assigned to each document
            - topic_info: DataFrame with information about each topic
            - topic_representations: Dict with topic representations (words and their weights)
    """
    try:
        # Preprocess texts to remove stopwords
        preprocessed_texts = []
        for text in texts:
            preprocessed_text = preprocess_text_for_n_grams(text)
            if preprocessed_text.strip():  # Only add non-empty texts
                preprocessed_texts.append(preprocessed_text)
        
        # Skip if there are not enough texts after preprocessing
        if len(preprocessed_texts) < 5:
            return {
                "error": "Not enough text documents after preprocessing",
                "topics": [],
                "topic_info": [],
                "topic_representations": {}
            }
        
        # Initialize BERTopic model with improved configuration
        topic_model = BERTopic()
        
        # Fit the model and transform texts to get topics
        topics, probs = topic_model.fit_transform(preprocessed_texts)
        
        # Get topic info and representation
        topic_info = topic_model.get_topic_info()
        
        # Get the topic representations (words and their weights)
        topic_representations = {}
        for topic_id in set(topics):
            if topic_id != -1:  # Skip outlier topic
                topic_words = topic_model.get_topic(topic_id)
                topic_representations[str(topic_id)] = topic_words
        
        # Return results
        result = {
            "topics": topics,
            "topic_info": topic_info.to_dict('records'),
            "topic_representations": topic_representations
        }
        
        return result
    except Exception as e:
        print(f"Error in topic extraction: {str(e)}")
        return {
            "error": str(e),
            "topics": [],
            "topic_info": [],
            "topic_representations": {}
        }


async def get_post_title_and_description(post):
    try:
        flattened_post_text = str(post.title) + "\n" + str(post.selftext)
        upvotes = post.score
        return (flattened_post_text, upvotes)
    except:
        print('could not get post title and description')


def get_readability_metrics(posts):
    total_num_words_title = 0
    total_num_words_description = 0
    num_posts_over_100_words = 0
    readability_metrics = {
        "flesch_grade_level": 0.0,
        "dale_chall_grade_levels": defaultdict(int)
    }
    for post in posts:
        num_words_title = len(post.title.split())
        num_words_description = len(post.description.split())
        total_num_words_title += num_words_title
        total_num_words_description += num_words_description
        if num_words_description >= 100:
            num_posts_over_100_words += 1
            try:
                r = Readability(post.description)
                flesch_kincaid = r.flesch_kincaid()
                readability_metrics["flesch_grade_level"] = float(flesch_kincaid.grade_level) + float(readability_metrics.get("flesch_grade_level", 0))
                dale_chall = r.dale_chall()
                for level in dale_chall.grade_levels:
                    readability_metrics["dale_chall_grade_levels"][level] += 1
            except:
                print("An error occurred while generating readability metrics")
    avg_num_words_title = total_num_words_title / len(posts) if len(posts) != 0 else None
    avg_num_words_description = total_num_words_description / len(posts) if len(posts) != 0 else None
    avg_flesch_grade_level =  readability_metrics.get("flesch_grade_level", 0) / num_posts_over_100_words if num_posts_over_100_words != 0 else -1
    return {
             "avg_num_words_title": avg_num_words_title,
             "avg_num_words_description": avg_num_words_description,
             "avg_flesch_grade_level": avg_flesch_grade_level,
             "dale_chall_grade_levels": readability_metrics["dale_chall_grade_levels"]
            }
    
# posts_list is a list of RedditPost objects
# function returns a sorted map by date where:
#   key = date of slice (Ex: 07/24)
#   value = vector of RedditPost objects for slice 

def slice_posts_list(posts_list):
    slice_to_posts = dict()
    time_filter = config['time_filter']
    for post in posts_list:
        # convert post date from utc to readable format 
        date_format = ''
        if time_filter == 'all': date_format = '%y' # slicing all time by years 
        elif time_filter == 'year': date_format = '%m-%y' # slicing a year by months 
        else: date_format = '%m-%d' # slicing a month and week by days 
        post_date = datetime.fromtimestamp(post.created_utc).strftime(date_format)

        if post_date not in slice_to_posts:
            slice_to_posts[post_date] = []
        slice_to_posts[post_date].append(post)

    # sort slice_to_posts by date
    dates = list(slice_to_posts.keys())
    sorted_months = sorted(dates, key=lambda x: datetime.strptime(x, date_format))
    sorted_slice_to_posts = {i: slice_to_posts[i] for i in sorted_months}
    return sorted_slice_to_posts

async def perform_subreddit_analysis(subreddit_query: SubredditQuery):
    # !!! WARNING !!!
    # sometimes PRAW will be down, or the # of posts you're trying to get at once 
    # is too much. PRAW will often return a 500 HTTP response in this case.
    # Try to: reduce post_limit, or wait a couple minutes

    # Lol
    def set_progress(_bruh_):
        pass
    
    config['time_filter'] = subreddit_query.time_filter

    reddit = asyncpraw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="reddit sampler",
    )

    initial_post_query_limit = TIME_FILTER_TO_INITIAL_POST_QUERY_LIMIT[subreddit_query.time_filter]

    subreddit_instance = await reddit.subreddit(subreddit_query.name)
    posts = [post async for post in subreddit_instance.top(
        limit=initial_post_query_limit,
        time_filter=subreddit_query.time_filter
    )]
    print("initially queried ", len(posts), posts)

    # Get only the posts with a text description (throwing away the posts where the description is a link or photo)
    # high_text_posts = [post for post in posts if len(post.selftext) > 0]
    # print("len(high_text_posts): ", len(high_text_posts))
    if(len(posts) > TIME_FILTER_TO_POST_LIMIT[subreddit_query.time_filter]):
        # Get the posts with the most amount of description text 
        posts.sort(key=lambda post: len(post.selftext) + len(post.title))
        posts = posts[-TIME_FILTER_TO_POST_LIMIT[subreddit_query.time_filter]:]

    # if the number of posts is >= 200 and <= 300, then divide the posts list in half and wait a minute in between 
    # querying the comments for the 1st batch and the 2nd batch (to avoid going over asyncpraw's api limit)
    # if the number of posts is >= 300, then divide the posts list into thirds and wait a minute in between 
    # querying the comments for the 1st batch, 2nd batch, and 3rd batch (to avoid going over asyncpraw's api limit)
    post_batches = []
    if len(posts) >= 200 and len(posts) <= 300:
        mid = len(posts) // 2
        post_batches.append(posts[:mid])
        post_batches.append(posts[mid:])
    elif len(posts) > 300:
        third = len(posts) // 3
        post_batches.append(posts[:third])
        post_batches.append(posts[third:2*third])
        post_batches.append(posts[2*third:])


    posts_list = []
    if len(post_batches) > 0:
        for post_batch in post_batches:
            # Fetch post data concurrently
            posts_with_comments = await asyncio.gather(*(fetch_post_data(post) for post in post_batch))
            posts_with_comments = [post for post in posts_with_comments if post is not None]
            posts_list.extend(posts_with_comments)
            print("got the comments for ", len(posts_with_comments), "/", len(post_batch), " of the posts") 
            time.sleep(65)  # Wait for 65 seconds...
    else:
        posts_list = await asyncio.gather(*(fetch_post_data(post) for post in posts))
    print("# of posts after fetching comments: ", len(posts_list))

    # Find out how many words we're analyzing in total 
    total_words = 0
    for post in posts_list:
        total_words += len(post.title.split()) + len(post.description.split()) + sum([len(comment.split()) for comment in post.comments])
    print('total_words: ', total_words)

    # Save post to file (uncomment to use)
    # await print_to_json(posts_list, "posts.txt")

    sorted_slice_to_posts = slice_posts_list(posts_list)
    print('Finished getting sorted_slice_to_posts')
    print('got posts from the slices: ', sorted_slice_to_posts.keys())
    
    readability_metrics = get_readability_metrics(posts_list)
    print('Finished getting readability_metrics')
    print(readability_metrics)

    #plot_post_distribution(subreddit, time_filter, sorted_slice_to_posts)
    top_n_grams, top_named_entities = await get_subreddit_analysis(sorted_slice_to_posts, set_progress)
    print(top_named_entities)

    t1 = time.time()
    entity_set= set()
    for _, entities in top_named_entities.items():
        entity_names = [entity[0] for entity in entities] # getting just the name of each entity 
        for entity_name in entity_names: entity_set.add(entity_name)
    print('# of elements in entity_set: ', len(entity_set))
    top_named_entities_embeddings = get_2d_embeddings(list(entity_set))
    print("# of entity embeddings: ", len(top_named_entities_embeddings))
    print(top_named_entities_embeddings)
    t2 = time.time()
    print('getting 2d embeddings took: ', t2 - t1)
    # toxicity_score, toxicity_grade, toxicity_percentile, all_toxicity_scores, all_toxicity_grades = await get_toxicity_metrics(subreddit_query.name)
    # positive_content_score, positive_content_grade, positive_content_percentile, all_positive_content_scores, all_positive_content_grades = await get_positive_content_metrics(subreddit_query.name)
    print('generating wordcloud: ')
    top_named_entities_wordcloud = generate_word_cloud(top_named_entities)
    print("wordcloud generated")
    
    analysis = SubredditAnalysis(
        timestamp = int(time.time()),
        num_words = total_words,
        subreddit = subreddit_query.name,
        top_n_grams = top_n_grams,
        top_named_entities = top_named_entities,
        top_named_entities_embeddings = top_named_entities_embeddings,
        top_named_entities_wordcloud = top_named_entities_wordcloud,
        readability_metrics =  readability_metrics,   
    )

    return analysis
