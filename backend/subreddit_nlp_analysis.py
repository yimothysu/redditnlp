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
import asyncpraw
from dotenv import load_dotenv
from detoxify import Detoxify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import percentileofscore


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

def get_sentiment_of_entity(entity, flattened_sentences):
    t1 = time.time()
    sentiment = classifier(flattened_sentences, text_pair=entity)
    t2 = time.time()
    print('getting sentiment of the entity ', entity, ' took ', t2 - t1)
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

def get_subreddit_topics(sorted_slice_to_posts, num_topics=10):
    """
    Extract topics from subreddit posts and comments for each time slice.
    
    Args:
        sorted_slice_to_posts (dict): Dictionary mapping time slices to posts
        num_topics (int, optional): Number of topics to extract per slice. Defaults to 10.
        
    Returns:
        dict: Dictionary mapping time slices to topic extraction results
    """
    t1 = time.time()
    subreddit_topics = {}
    
    for date, posts in sorted_slice_to_posts.items():
        # Combine post titles and comments for topic modeling
        texts = []
        for post in posts:
            # Add post title (titles are often more informative than comments)
            if post.title and len(post.title.strip()) > 10:  # Only add substantial titles
                texts.append(post.title)
            
            # Process and add comments
            filtered_comments = []
            for comment in post.comments:
                # Skip very short comments or empty comments
                if comment and len(comment.strip()) > 15:
                    filtered_comments.append(comment)
            
            # Add filtered comments
            texts.extend(filtered_comments)
        
        # Skip if there are not enough texts for meaningful topic modeling
        if len(texts) < 5:
            continue
            
        # Extract topics from the texts
        topics_result = extract_topics(texts, num_topics)
        subreddit_topics[date] = topics_result
    
    t2 = time.time()
    print('finished topic extraction in: ', t2-t1)
    
    return subreddit_topics


def get_percentile(values, target_value):
    # gets the percentile of target_value in values distribution 
    percentile = percentileofscore(values, target_value, kind='rank')
    return round(percentile, 4)

def get_toxicity_grade(toxicity):
    # A+ means extremely civil, respectful, and non-toxic subreddit.
    # F- means extremely toxic subreddit 
    toxicity *= 100 # convert toxicity from [0, 1] scale to [0, 100] scale
    grade_to_toxicity_cutoff = {'A+': 0.05, 'A': 0.1, 'B+': 0.5, 'B': 1, 'B-': 5, 
                                'C+': 10, 'C': 15, 'C-': 20, 'D+': 30, 'D': 40, 'D-': 50}
    for grade, toxicity_cutoff in grade_to_toxicity_cutoff.items():
        if toxicity <= toxicity_cutoff: return grade
    return "F"

def get_positive_content_grade(positive_content_score):
    # A+ means very positive content subreddit
    # F means very negative content subreddit 
    positive_content_score *= 100 # convert positive_content from [0, 1] scale to [0, 100] scale
    grade_to_positive_content_cutoff = {'A+': 92, 'A': 84, 'B+': 75, 'B': 68, 'B-': 60, 'C+': 52,
                                        'C': 43, 'C-': 35, 'D+': 26, 'D': 17, 'D-': 9}
    for grade, positive_content_cutoff in grade_to_positive_content_cutoff.items():
        if positive_content_score >= positive_content_cutoff: return grade
    return "F"


async def get_post_title_and_description(post):
    try:
        flattened_post_text = str(post.title) + "\n" + str(post.selftext)
        upvotes = post.score
        return (flattened_post_text, upvotes)
    except:
        print('could not get post title and description')


def composite_toxicity(text):
    weights = {
        'toxicity': 1.0,
        'severe_toxicity': 1.5,  
        'obscene': 0.8,
        'threat': 1.5,  
        'insult': 1.0,
        'identity_attack': 1.2
    }
    scores = Detoxify('original').predict(text)
    total_weight = sum(weights.values())
    composite_toxicity_score = round(sum(scores[k] * weights[k] for k in scores) / total_weight, 6)
    return composite_toxicity_score

# This function will compute and return 4 values:
#   1. toxicity_score: float # [0, 1] --> 0 = not toxic at all, 1 = all toxic 
#   2. toxicity_grade: str # A+ to F 
#   3. toxicity_percentile: float # [0, 100]
#   4. all_toxicity_scores: List[float] # for generating a toxicity distribution graph on the UI 
async def get_toxicity_metrics(sub_name):
    toxicity_dict = {}
    with open("top_subreddits_toxicity_score.txt", "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" ") 
            if len(parts) == 2: 
                subreddit_name, score = parts
                toxicity_dict[subreddit_name] = round(float(score), 4)
    all_toxicity_scores = list(toxicity_dict.values())
    
    if (sub_name in toxicity_dict):
        print('toxicity score already in top_subreddits_toxicity_score.txt')
        toxicity_score = toxicity_dict[sub_name]
        return (toxicity_score, get_toxicity_grade(toxicity_score), get_percentile(all_toxicity_scores, toxicity_score), all_toxicity_scores)
    else:
        reddit = asyncpraw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="reddit_api"
        )
        try:
            subreddit_instance = await reddit.subreddit(sub_name)
            posts = [post async for post in subreddit_instance.top(
                limit=400,
                time_filter="all"
            )]
            posts = await asyncio.gather(*(get_post_title_and_description(post) for post in posts))
            posts = [post for post in posts if post is not None]

            all_flattened_post_text = "\n".join([flattened_post_text for flattened_post_text, _ in posts])
            composite_toxicity_score = composite_toxicity(all_flattened_post_text)
            await reddit.close() 
            
            f = open("top_subreddits_toxicity_score.txt", "a", encoding="utf-8")
            f.write(sub_name + " " + str(composite_toxicity_score) + "\n")
            print('wrote toxicity score for r/', sub_name, ' to top_subreddits_toxicity_score.txt')
            all_toxicity_scores.append(composite_toxicity_score)
            return (composite_toxicity_score, get_toxicity_grade(composite_toxicity_score), get_percentile(all_toxicity_scores, composite_toxicity_score), all_toxicity_scores)
        except Exception as e:
            print('couldnt get posts to calculate toxicity score because ', e)


# This function will compute and return 4 values:
#   1. positive_content_score: float # [0, 1] --> 0 = no positive content, 1 = all positive content
#   2. positive_content_grade: str # A+ to F 
#   3. positive_content_percentile: float # [0, 100]
#   4. all_positive_content_scores: List[float] # for generating a positive content scores distribution graph on the UI 
async def get_positive_content_metrics(sub_name):
    positive_content_dict = {}
    with open("top_subreddits_positive_content_score.txt", "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" ") 
            if len(parts) == 2: 
                subreddit_name, score = parts
                positive_content_dict[subreddit_name] = round(float(score), 4)
    all_positive_content_scores = list(positive_content_dict.values())
    all_positive_content_scores_normalized = [(x + 1) / 2 for x in all_positive_content_scores] # changes score range from [-1, 1] to [0, 1]
    positive_content_score = 0
    
    if (sub_name in positive_content_dict):
        print('positive content score already in top_subreddits_positive_content_score.txt')
        positive_content_score = positive_content_dict[sub_name]
        positive_content_score_normalized = (positive_content_score + 1) / 2 # changes score range from [-1, 1] to [0, 1]
        return (positive_content_score_normalized, 
                get_positive_content_grade(positive_content_score_normalized), 
                get_percentile(all_positive_content_scores_normalized, positive_content_score_normalized), 
                all_positive_content_scores_normalized)
    else:
        reddit = asyncpraw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="reddit_api"
        )
        analyzer = SentimentIntensityAnalyzer()

        try:
            subreddit_instance = await reddit.subreddit(sub_name)
            posts = [post async for post in subreddit_instance.top(
                limit=400,
                time_filter="all"
            )]
            posts = await asyncio.gather(*(get_post_title_and_description(post) for post in posts))
            posts = [post for post in posts if post is not None]

            weighted_sentiment_sum = 0
            total_upvotes = 0
            for flattened_post_text, upvotes in posts:
                score = analyzer.polarity_scores(flattened_post_text)
                weighted_sentiment_sum += score['compound'] * upvotes
                total_upvotes += upvotes
            positive_content_score = weighted_sentiment_sum / total_upvotes
            positive_content_score_normalized = (positive_content_score + 1) / 2
            await reddit.close() 

            f = open("top_subreddits_positive_content_score.txt", "a", encoding="utf-8")
            f.write(sub_name + " " + str(positive_content_score) + "\n")
            print('wrote positive content score for r/', sub_name, ' to top_subreddits_positive_content_score.txt')
            all_positive_content_scores_normalized.append(positive_content_score_normalized)
            return (positive_content_score_normalized, 
                    get_positive_content_grade(positive_content_score_normalized), 
                    get_percentile(all_positive_content_scores_normalized, positive_content_score_normalized), 
                    all_positive_content_scores_normalized)
        except Exception as e:
            print('couldnt get posts to calculate positive content score because ', e)