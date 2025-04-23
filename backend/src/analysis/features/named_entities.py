"""Module for extracting and analyzing named entities from subreddit content.

Provides functionality for:
- Named entity recognition using spaCy
- Sentiment analysis using ABSA (Aspect-Based Sentiment Analysis)
- Key point extraction using BART summarization
- Entity consolidation and filtering
"""

import time, asyncio, random, re, math, nltk, numbers  # type: ignore
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy  # type: ignore
from src.utils.subreddit_classes import ( NamedEntity )

# Load NLP models
absa_model_name = "yangheng/deberta-v3-base-absa-v1.1"
absa_model = AutoModelForSequenceClassification.from_pretrained(absa_model_name)
absa_tokenizer = AutoTokenizer.from_pretrained(absa_model_name)
classifier = pipeline("text-classification", model=absa_model, tokenizer=absa_tokenizer)
summarizer= pipeline("summarization", model="facebook/bart-large-cnn")
summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Configuration constants
TIME_FILTER_TO_NAMED_ENTITIES_LIMIT = {'all': 12, 'year': 10, 'week': 8}
TIME_FILTER_TO_KEY_POINTS_LIMIT = {'week': 3, 'month': 3, 'year': 5, 'all': 7}
TIME_FILTER_TO_POST_URL_LIMIT = {'week': 1, 'year': 3, 'all': 3}

config = {
    "time_filter": "week",
    "max_absa_tokens": 2000,
    "max_input_len_to_summarizer": 2500, # in characters 
}

banned_entities = {'one', 'two', 'first', 'second', 'yesterday', 'today', 'tomorrow', 'approx', 'half', 'idk', 'congrats', 'three', 'creepy', 'night',
                   'day', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'month', 'months', 'day', 'days', 'week',
                   'weeks', 'this year', 'next year', 'last year', 'year', 'a year', 'years', 'a couple years', 'the week', 'the summer', 'spring', 
                   'summer', 'fall', 'winter', 'zero', 'one day', 'the next day', 'six', 'seven', 'eight', 'the years', 'every day', "n't", 'that day',
                   'one year', 'each day', 'last week', 'all day', 'a couple of weeks', 'daily', 'some days', 'hours', 'a week', 'some days', 'the day', 
                   'a few years', 'every morning', 'morning', 'an hour', 'a month', 'a year ago', 'most days', 'tonight', 'overnight', 'the end of the day', 
                   'kinda', 'last month', 'a few minutes', 'dude', 'those years', 'n’t', 'lot', 'only two', 'ten', 'the morning', 'five years', 'op', 'a ton', 
                   'a few years ago', 'decades', 'a few months', 'a few days', 'over a decade', 'this summer', 'a bad day', 'the days', 'these days', 'shit', 
                   'fuck', 'ass', 'asshole', 'bitch', 'three years', 'two years', 'one year', 'every single day', 'this day', 'thousands', 'af', 'weekly', 'a decade', 
                   'another day', 'each year', 'each week', 'ten years', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 
                   'october', 'november', 'december', 'four', 'five', 'nine', 'this week', 'next week', 'a few more years', 'a new day', 'the other day', 'third', 
                   'every night', 'max', 'nah', 'yeah', 'yea', 'rip', 'last night', 'six months', 'years later', 'about a week', 'more than one', 'depends', 
                   'about a year', 'every night', 'monthly', 'each month', 'one night', 'the night', 'the year', 'a few weeks', 'nights', 'that night', 'a few hours', 
                   'that year', 'the first year', 'time', 'a couple of years', 'a nice day', 'reader', 'nsfw', 'youtube'}


# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('vader_lexicon')

async def get_top_entities(time_filter, post_content_grouped_by_date, posts_grouped_by_date):
    """Extract and analyze top named entities from grouped post content.
    
    Args:
        time_filter: Time period filter ('all', 'year', 'week')
        post_content_grouped_by_date: Dictionary mapping dates to post content
        posts_grouped_by_date: Dictionary mapping dates to post objects
    Returns:
        dict: Mapping of dates to lists of analyzed NamedEntity objects
    """
    config["time_filter"] = time_filter 
    dates = list(post_content_grouped_by_date.keys())
    post_content = list(post_content_grouped_by_date.values())
    
    # for each entity, get: 
    #    - sentiment score
    #    - key points
    #    - top post urls
    t1 = time.time()
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
    nlp.add_pipe('sentencizer')
    top_named_entities = dict()
    for date, doc in zip(dates, nlp.pipe(post_content, batch_size=50)):
        top_named_entities[date] = doc
    t2 = time.time()
    print('Computing top named entities using nlp.pipe took: ', t2-t1)
    
    # Process entities in parallel
    tasks = [asyncio.create_task(postprocess_named_entities(date, doc)) for _, (date, doc) in enumerate(top_named_entities.items())]
    results = await asyncio.gather(*tasks)
    for date, entities in results:
        top_named_entities[date] = entities
    top_named_entities = get_post_urls(posts_grouped_by_date, top_named_entities)
    t3 = time.time()
    print('NER analysis took: ', t3-t2)
    return top_named_entities 

def filter_named_entity(name: str) -> bool:
    """Check if a string is a valid named entity.
    
    Validates that the entity name consists of proper words,
    allowing for hyphenations and apostrophes.
    
    Args:
        name: String to validate
    Returns:
        bool: True if valid named entity, False otherwise
    """
    parts = name.split()
    if len(parts) == 0: return False

    # Check if each word is indeed a word, allowing for hyphenations and apostrophes
    word_pattern = re.compile(r"^[a-zA-Z]+([-'’][a-zA-Z]+)*$")
    for part in parts:
        if not word_pattern.match(part):
            return False
    return True


async def postprocess_named_entities(date, doc):
    """Process and analyze named entities from a spaCy document.
    
    Filters entities, calculates frequencies, and enriches with
    sentiment and key points.
    
    Args:
        date: Date string for the document
        doc: spaCy Doc object
    Returns:
        tuple: (date, list of processed NamedEntity objects)
    """
    # Count and filter entities
    entities = Counter([ent.text for ent in doc.ents])
    filtered_entities = Counter()
    for name, count in entities.items():
        if count < 3: continue # not enough sentences to form a meaningful analysis of the entity 
        if not (isinstance(name, str) and filter_named_entity(name)): continue
        if isinstance(name, numbers.Number) or name.isnumeric() or name.lower().strip() in banned_entities: continue
        filtered_entities[name] = count

    # Get top entities and analyze
    top_entities = filtered_entities.most_common(TIME_FILTER_TO_NAMED_ENTITIES_LIMIT[config['time_filter']])
    top_entity_names = set([entity[0] for entity in top_entities])
    t1 = time.time()
    entity_to_sentiment, entity_to_key_points = await get_sentiment_and_key_points_of_top_entities(top_entity_names, doc)
    t2 = time.time()
    print('Computing sentiment + key points of named entities in ', date, ' took: ', t2 - t1)

    for i in range(len(top_entities)):
        entity = NamedEntity(
            name = top_entities[i][0],
            count = top_entities[i][1]
        )
        if entity.name in entity_to_sentiment and entity.name in entity_to_key_points:
            entity.sentiment = round(entity_to_sentiment[entity.name], 2)
            entity.key_points = entity_to_key_points[entity.name]
            top_entities[i] = entity 
        else:
            top_entities[i] = None
    top_entities = [top_entity for top_entity in top_entities if top_entity is not None]
    return (date, top_entities)


async def get_sentiment_and_key_points_of_top_entities(top_entity_names, doc):
    """Get sentiment and key points for top entities from a spaCy document.
    
    Args:
        top_entity_names: List of entity names to analyze
        doc: spaCy Doc object
    Returns:
        entity_to_sentiment: Dictionary mapping entity names to sentiment scores
        entity_to_key_points: Dictionary mapping entity names to key points
    """
    entity_to_sentiment  = dict()
    entity_to_sentences = dict()
    entity_to_flattened_sentences = dict()
    entity_to_key_points = dict() 
    for ent in doc.ents:
        ent_text = ent.text 
        if ent_text in top_entity_names:
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
        entity_to_flattened_sentences[entity] = flattened_sentences
    
    tasks = [asyncio.to_thread(get_sentiment_of_entity, entity, entity_to_flattened_sentences[entity]) for entity in entity_to_sentences]
    sentiment_results = await asyncio.gather(*tasks)
    sentiment_results = [sentiment_result for sentiment_result in sentiment_results if sentiment_result is not None]
    for entity, sentiment_score in sentiment_results:
        entity_to_sentiment[entity] = sentiment_score
    
    tasks = [asyncio.to_thread(get_key_points_of_entity, entity, entity_to_flattened_sentences[entity]) for entity in entity_to_sentences]
    key_points_results = await asyncio.gather(*tasks)
    key_points_results = [result for result in key_points_results if result is not None]
    for entity, key_points in key_points_results:
        entity_to_key_points[entity] = key_points
    
    entity_to_sentiment, entity_to_key_points = combine_same_entities(entity_to_sentiment, entity_to_key_points)
    return entity_to_sentiment, entity_to_key_points


def combine_two_entities(entity_to_sentiment, entity_to_key_points, entity_1, entity_2):
    """For 2 entities that refer to the same thing but have a slightly different spelling (Ex: US, USA), MERGE the 2 entities.
        - Take the average of their sentiment scores 
        - Combine their key points
    """
    print('combining ', entity_1, ' and ', entity_2)
    combined_sentiment = (entity_to_sentiment[entity_1] + entity_to_sentiment[entity_2]) / 2
    entity_to_sentiment[entity_1] = combined_sentiment
    del entity_to_sentiment[entity_2]

    combined_key_points = "\n".join(entity_to_key_points[entity_1]) + " \n " + "\n".join(entity_to_key_points[entity_2])
    _, combined_key_points = get_key_points_of_entity(entity_1, combined_key_points)
    entity_to_key_points[entity_1] = combined_key_points
    del entity_to_key_points[entity_2]

    return entity_to_sentiment, entity_to_key_points


def combine_same_entities(entity_to_sentiment, entity_to_key_points):
    """Combine entities that refer to the same thing
    Ex: (republican, republicans), (Democrat, democrats), (US, USA, America), (American, Americans) 
    Any pair of entities that are the same if you lowercase the first letter —> Ex: China, china
    """
    print('inside combine_same_entities')
    for entity in list(entity_to_key_points.keys()):
        entity_lowercase = entity[0].lower() + entity[1:]
        entity_lowercase_plural = entity[0].lower() + entity[1:] + "s"
        entity_plural = entity + "s"
        
        if entity_lowercase in entity_to_key_points and entity != entity_lowercase:
            entity_to_sentiment, entity_to_key_points = combine_two_entities(entity_to_sentiment, entity_to_key_points, entity, entity_lowercase)
        
        if entity_lowercase_plural in entity_to_key_points:
            entity_to_sentiment, entity_to_key_points = combine_two_entities(entity_to_sentiment, entity_to_key_points, entity, entity_lowercase_plural)
        
        if entity_plural in entity_to_key_points:
            entity_to_sentiment, entity_to_key_points = combine_two_entities(entity_to_sentiment, entity_to_key_points, entity, entity_plural)

        america_synonyms = ["USA", "US", "America", "america", "usa"]
        if entity in america_synonyms and entity in entity_to_key_points:
            america_synonyms.remove(entity)
            sentiments = [entity_to_sentiment[entity]]
            key_points = ['\n'.join(entity_to_key_points[entity])]
            
            for america_synonym in america_synonyms:
                if america_synonym in entity_to_key_points:
                    sentiments.append(entity_to_sentiment[america_synonym])
                    key_points.append('\n'.join(entity_to_key_points[america_synonym]))
                    del entity_to_sentiment[america_synonym]
                    del entity_to_key_points[america_synonym]
            
            if len(sentiments) > 1:
                print('combining ', entity, ' with ', len(sentiments), ' synonyms')
                combined_sentiment = sum(sentiments) / len(sentiments)
                _, combined_key_points = get_key_points_of_entity(entity, ' \n '.join(key_points))
                entity_to_sentiment[entity] = combined_sentiment
                entity_to_key_points[entity] = combined_key_points

    return entity_to_sentiment, entity_to_key_points


def split_string_into_chunks(text, num_chunks):
    """Split text into roughly equal chunks for processing.
    
    Args:
        text: Text to split
        num_chunks: Number of chunks to create
    Returns:
        list: Non-empty text chunks
    """
    words = text.split()
    avg = math.ceil(len(words) / num_chunks)
    chunks = [" ".join(words[i * avg:(i + 1) * avg]) for i in range(num_chunks)]
    return [chunk for chunk in chunks if chunk]


def get_sentiment_of_entity(entity, flattened_sentences):
    """Calculate sentiment score for an entity using ABSA.
    
    Handles long texts by splitting into chunks and averaging scores.
    
    Args:
        entity: Entity name to analyze
        flattened_sentences: Combined sentences mentioning the entity
    Returns:
        tuple: (entity name, sentiment score) or None if analysis fails
    """
    absa_tokenizer = AutoTokenizer.from_pretrained(absa_model_name)
    tokens = absa_tokenizer(flattened_sentences, truncation=False)
    if(len(tokens['input_ids']) >= config['max_absa_tokens']):
        t1 = time.time()
        num_chunks = math.ceil(len(tokens['input_ids']) / 2000)
        chunks = split_string_into_chunks(flattened_sentences, num_chunks)
        averaged_sentiment_score = 0
        
        for chunk in chunks:
            tokens = absa_tokenizer(chunk, truncation=False)
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


def get_key_points_of_entity(entity, flattened_comments):
    """Get the key points of an entity from a list of comments."""
    summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    key_points_limit = TIME_FILTER_TO_KEY_POINTS_LIMIT[config['time_filter']]
    if len(flattened_comments) > config['max_input_len_to_summarizer']:
        # highly likely to exceed facebook/bart-large-cnn token limit of 1024, so analyze each chunk seperately
        t1 = time.time()
        chunks = [flattened_comments[i:i + config['max_input_len_to_summarizer']] for i in range(0, len(flattened_comments), config['max_input_len_to_summarizer'])]
        key_points = []
        for chunk in chunks:
            tokenized = summarizer_tokenizer.encode(chunk, truncation=True, max_length=1024)
            truncated_chunk = summarizer_tokenizer.decode(tokenized)
            model_output = summarizer(truncated_chunk, max_length=100, min_length=20, do_sample=False)
            model_summary_text = model_output[0]['summary_text']
            relevant_sentences = [sentence for sentence in nltk.sent_tokenize(model_summary_text) if entity in sentence] # Only keep the sentences in summary_text which mention the entity 
            key_points.extend(relevant_sentences)
        if len(key_points) > key_points_limit and len(key_points) <= key_points_limit + 5:
            key_points = random.sample(key_points, key_points_limit)
        elif len(key_points) > key_points_limit + 5: # happens for an entity that is mentioned in A LOT of comments 
            _, key_points = get_key_points_of_entity(entity, "\n".join(key_points))
        t2 = time.time()
        print('Computing key points of ', entity, ' took: ', t2 - t1)
        return (entity, key_points)
    else:
        t1 = time.time()
        tokenized = summarizer_tokenizer.encode(flattened_comments, truncation=True, max_length=1024)
        truncated_flattened_comments = summarizer_tokenizer.decode(tokenized) 
        model_output = summarizer(truncated_flattened_comments, max_length=100, min_length=5, do_sample=False)
        model_summary_text = model_output[0]['summary_text']
        key_points = [sentence for sentence in nltk.sent_tokenize(model_summary_text) if entity in sentence] # Only keep the sentences in summary_text which mention the entity 
        if len(key_points) > key_points_limit:
            key_points = random.sample(key_points, key_points_limit)
        t2 = time.time()
        print('Computing key points of ', entity, ' took: ', t2 - t1)
        return (entity, key_points)


def get_post_urls(date_to_posts, date_to_entities):
    """Get the top post urls for each named entity so users can reference the posts."""
    if date_to_posts == {}:
        return date_to_entities
    t1 = time.time()
    num_urls_max = TIME_FILTER_TO_POST_URL_LIMIT[config['time_filter']]
    new_date_to_entities = {}
    for date, entities in date_to_entities.items():
        for entity in entities:
            posts_that_mention_entity = [] # idxs of the posts 
            for post in date_to_posts[date]:
                post_content = post.title + ' ' + post.description + ' '.join(post.comments)
                if entity.name in post_content: 
                    posts_that_mention_entity.append((post.url, post_content.count(entity.name), post.score))
            # Time to pick which posts to feature 
            # sort posts_that_mention_entity by post_content.count(entity_name) * score 
            posts_that_mention_entity.sort(key=lambda post: post[1] * post[2])
            post_urls_for_entity = [post[0] for post in posts_that_mention_entity[-num_urls_max:]]
            if date not in new_date_to_entities:
                new_date_to_entities[date] = []
            entity.urls = post_urls_for_entity
            new_date_to_entities[date].append(entity)
    t2 = time.time()
    print('get_post_urls took: ', t2 - t1)
    return new_date_to_entities
