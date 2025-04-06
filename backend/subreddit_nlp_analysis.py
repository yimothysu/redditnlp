import numbers 
import re 
from collections import Counter
from pydantic import BaseModel
from typing import Dict, List, Tuple 
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import asyncio
import re
import os 
import math 
import random 
import asyncpraw
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timezone
from readability import Readability

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
from spacy.tokens import Doc

from reddit_topic_extractor import (
    fetch_post_data,
)

from subreddit_classes import (
    SubredditQuery,
    SubredditAnalysis,
)

from word_embeddings import (
    get_2d_embeddings,
)

from word_cloud import (
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
model_name = "yangheng/deberta-v3-base-absa-v1.1"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
summarizer= pipeline("summarization", model="facebook/bart-large-cnn")

TIME_FILTER_TO_POST_LIMIT = {'all': 400, 'year': 200, 'month': 100, 'week': 50}

load_dotenv()

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
    vectors = [model[word] for word in words if word in model]
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
    
    # Perform n-gram analysis - just only take ~ 1-2 seconds 
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

    # Perform NER analysis + sentiment analysis to EACH named entity 
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
    nlp.add_pipe('sentencizer')
    t3 = time.time()
    print('initializing nlp took: ', t3 - t2)

    top_named_entities = dict()
    for date, doc in zip(dates, nlp.pipe(texts, batch_size=50)):
        top_named_entities[date] = doc
    # for i in range(len(texts)):
    #     text = texts[i]
    #     date = dates[i]
    #     if len(text) >= 1000000:
    #          text_chunks = [text[i:i + 800000] for i in range(0, len(text), 800000)]
    #          print("num text chunks: ", len(text_chunks))
    #          all_docs = []
    #          for text_chunk in text_chunks:
    #              doc = nlp.pipe(text_chunk, batch_size=50)
    #              all_docs.extend(list(doc))
    #          all_docs = [doc for doc in all_docs if doc is not None]
    #          if all_docs is not None:
    #             combined_doc = Doc.from_docs(all_docs)
    #             # Convert back to generator
    #             combined_doc_gen = (doc for doc in combined_doc)
    #             top_named_entities[date] = combined_doc_gen
    #     else:
    #         print("num text chunks: 1")
    #         doc = nlp.pipe(text, batch_size=50)
    #         top_named_entities[date] = doc
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

    t5 = time.time()
    print('finished NER analysis in: ', t3-t2)
    
    return top_n_grams, top_named_entities

# async def postprocess_named_entities(date, doc, idx, set_progress):
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
                       'five', 'nine', 'this week', 'next week', 'a few more years', 'a new day'}
    # clustered_named_entites = cluster_similar_entities(named_entities)
    filtered_named_entities = Counter()
    for name, count in named_entities.items():
        if count < 3: continue # not enough sentences to sample 
        if not (isinstance(name, str) and filter_named_entity(name)): continue
        if isinstance(name, numbers.Number) or name.isnumeric() or name.lower().strip() in banned_entities: continue
        filtered_named_entities[name] = count
    t2 = time.time()

    top_ten_entities = filtered_named_entities.most_common(6)
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
        entity_sentiment = round(entity_to_sentiment[entity_name], 2)
        entity_sentences = entity_to_sentences[entity_name]
        top_ten_entities[i] = (entity_name, entity_count, entity_sentiment, entity_sentences)

    set_progress(1.0)

    return (date, top_ten_entities)

async def get_sentiments_of_entities(top_ten_entity_names, doc):
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

def split_string_into_chunks(text, num_chunks):
    words = text.split()
    avg = math.ceil(len(words) / num_chunks)
    chunks = [" ".join(words[i * avg:(i + 1) * avg]) for i in range(num_chunks)]
    return [chunk for chunk in chunks if chunk]

def get_sentiment_of_entity(entity, flattened_sentences):
    tokens = tokenizer(flattened_sentences, truncation=False)
    if(len(tokens['input_ids']) >= 2000):
        t1 = time.time()
        num_chunks = math.ceil(len(tokens['input_ids']) / 2000)
        chunks = split_string_into_chunks(flattened_sentences, num_chunks)
        averaged_sentiment_score = 0
        
        for chunk in chunks:
            tokens = tokenizer(chunk, truncation=False)
            print("Number of tokens:", len(tokens['input_ids']))
            sentiment = classifier(chunk, text_pair=entity)
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
        tokens = tokenizer(flattened_sentences, truncation=False)
        print("Number of tokens:", len(tokens['input_ids']))
        t1 = time.time()
        sentiment = classifier(flattened_sentences, text_pair=entity)
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


def split_text(text, chunk_size=3000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def summarize_comments(entity, flattened_comments):
    #print('# characters in text: ', len(text))
    if len(flattened_comments) > 3000:
        t1 = time.time()
        # will probably exceed the token limit of 1024, so analyze each 3000 character chunk seperately
        chunks = split_text(flattened_comments)
        complete_summary = ""
        for chunk in chunks:
            tokenized = tokenizer.encode(chunk, truncation=True, max_length=1024)
            truncated_chunk = tokenizer.decode(tokenized)
            summary = summarizer(truncated_chunk, max_length=100, min_length=20, do_sample=False)
            summary_text = summary[0]['summary_text']
            complete_summary += summary_text + "\n --- \n"
        t2 = time.time()
        print('getting summarized sentiment of entity (len(flattened_comments) > 3000)', entity, ' took ', t2 - t1)
        return (entity, complete_summary)
    else:
        t1 = time.time()
        tokenized = tokenizer.encode(flattened_comments, truncation=True, max_length=1024)
        truncated_flattened_comments = tokenizer.decode(tokenized) 
        num_tokens = len(tokenized)
        max_length = min(100, num_tokens)
        summary = summarizer(truncated_flattened_comments, max_length=max_length, min_length=5, do_sample=False)
        summary_text = summary[0]['summary_text']
        #print(summary_text)
        t2 = time.time()
        print('getting summarized sentiment of entity ', entity, ' took ', t2 - t1)
        return (entity, summary_text)


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
    readability_metrics = {}
    for post in posts:
        # num_words_title = len(word_tokenize(post.title))
        # num_words_description = len(word_tokenize(post.description))
        num_words_title = len(post.title.split())
        num_words_description = len(post.description.split())
        total_num_words_title += num_words_title
        total_num_words_description += num_words_description
        if num_words_description >= 100:
            num_posts_over_100_words += 1
            r = Readability(post.description)
            flesch_kincaid = r.flesch_kincaid()
            #dale_chall = r.dale_chall()
            readability_metrics["flesch_grade_level"] = float(flesch_kincaid.grade_level) + float(readability_metrics.get("flesh_grade_level", 0))
            #readability_metrics["dale_chall_grade_level"] = float(dale_chall.grade_levels) + float(readability_metrics.get("dale_chall_grade_level", 0))
    avg_num_words_title = total_num_words_title / len(posts) if len(posts) != 0 else None
    avg_num_words_description = total_num_words_description / len(posts) if len(posts) != 0 else None
    avg_flesh_grade_level =  readability_metrics.get("flesch_grade_level", 0) / num_posts_over_100_words if num_posts_over_100_words != 0 else -1
    #avg_dale_chall_grade_level = readability_metrics.get("dale_chall_score", 0) / num_posts_over_100_words if num_posts_over_100_words != 0 else -1
    return {
             "avg_num_words_title": avg_num_words_title,
             "avg_num_words_description": avg_num_words_description,
             "avg_flesh_grade_level": avg_flesh_grade_level,
             #"avg_dale_chall_grade_level": avg_dale_chall_grade_level
            }
    
# posts_list is a list of RedditPost objects
# function returns a sorted map by date where:
#   key = date of slice (Ex: 07/24)
#   value = vector of RedditPost objects for slice 

def slice_posts_list(posts_list, time_filter):
    slice_to_posts = dict()
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

    reddit = asyncpraw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="reddit sampler",
    )

    post_limit = TIME_FILTER_TO_POST_LIMIT[subreddit_query.time_filter]

    subreddit_instance = await reddit.subreddit(subreddit_query.name)
    posts = [post async for post in subreddit_instance.top(
        limit=post_limit,
        time_filter=subreddit_query.time_filter
    )]

    # Fetch post data concurrently
    posts_list = await asyncio.gather(*(fetch_post_data(post) for post in posts))
    posts_list = [post for post in posts_list if post is not None]

     # Find out how many words we're analyzing in total 
    total_words = 0
    for post in posts_list:
        total_words += len(post.title.split()) + len(post.description.split()) + sum([len(comment.split()) for comment in post.comments])
    print('total_words: ', total_words)

    # Save post to file (uncomment to use)
    # await print_to_json(posts_list, "posts.txt")

    sorted_slice_to_posts = slice_posts_list(posts_list, subreddit_query.time_filter)
    print('Finished getting sorted_slice_to_posts')
    
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
