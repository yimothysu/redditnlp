"""Module for extracting and analyzing named entities from subreddit content.

Provides functionality for:
- Named entity recognition using spaCy
- Sentiment analysis using ABSA (Aspect-Based Sentiment Analysis)
- Key point extraction using BART summarization
- Entity consolidation and filtering
"""

import time, asyncio, random, re, math, nltk, numbers, os, string  # type: ignore
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy  # type: ignore
from src.utils.subreddit_classes import ( NamedEntity, NamedEntityLabel )
from src.analysis.fix_pronoun_overresolution import ( make_text_natural_sounding )
from src.analysis.summarize_subreddit_comments import ( summarize_comments )
import hdbscan # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations

# Load NLP models
absa_model_name = "yangheng/deberta-v3-base-absa-v1.1"
absa_model = AutoModelForSequenceClassification.from_pretrained(absa_model_name)
absa_tokenizer = AutoTokenizer.from_pretrained(absa_model_name)
classifier = pipeline("text-classification", model=absa_model, tokenizer=absa_tokenizer)
summarizer= pipeline("summarization", model="facebook/bart-large-cnn")
summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Configuration constants
#TIME_FILTER_TO_NAMED_ENTITIES_LIMIT = {'all': 35, 'year': 30, 'week': 20}
#TIME_FILTER_TO_KEY_POINTS_LIMIT = {'week': 3, 'month': 3, 'year': 5, 'all': 7}
TIME_FILTER_TO_POST_URL_LIMIT = {'week': 1, 'year': 3, 'all': 3}
TIME_FILTER_TO_NUM_COMMENTS_PER_ENTITY_LIMIT = {'week': 100, 'year': 300, 'all': 500}
ALLOWED_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "WORK_OF_ART", "LAW", "EVENT", "LANGUAGE", "NORP"}

config = {
    "time_filter": "week",
    "max_absa_tokens": 2000,
    "max_input_len_to_summarizer": 2000, # in characters 
    "subreddit": "",
}

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('vader_lexicon')

nlp = spacy.load("en_core_web_trf", disable=["parser", "tagger"])
nlp.add_pipe('sentencizer')

dep_parsing_nlp = spacy.load("en_core_web_trf", disable=["tagger"])

async def get_top_entities(subreddit, time_filter, post_content_grouped_by_date, posts_grouped_by_date, comment_and_score_pairs_grouped_by_date):
    """Extract and analyze top named entities from grouped post content.
    
    Args:
        time_filter: Time period filter ('all', 'year', 'week')
        post_content_grouped_by_date: Dictionary mapping dates to post content
        posts_grouped_by_date: Dictionary mapping dates to post objects
    Returns:
        dict: Mapping of dates to lists of analyzed NamedEntity objects
    """
    config["subreddit"] = subreddit 
    config["time_filter"] = time_filter 
    dates = list(post_content_grouped_by_date.keys())
    post_content = list(post_content_grouped_by_date.values())
    post_content_safe = [] # safe meaning it won't exceed the models limit of 1000000 characters per text 
    # max length that post_content can be is 1000000 characters
    for post in post_content:
        if len(post) >= 950000:
            chunks = [post[i:i+950000] for i in range(0, len(post), 950000)]
            post_content_safe.extend(chunks)
            print("split a post's content into ", str(len(chunks)), " 950000 character chunks!")
        else:
            post_content_safe.append(post)

    t1 = time.time()
    top_named_entities = dict()
    for date, doc in zip(dates, nlp.pipe(post_content_safe, batch_size=50)):
        top_named_entities[date] = doc
    t2 = time.time()
    print('Computing top named entities using nlp.pipe took: ', t2-t1)

    for date, doc in top_named_entities.items():
        top_named_entities[date] = postprocess_named_entities(date, doc, comment_and_score_pairs_grouped_by_date[date])
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


def load_banned_entities():
    project_root = os.path.abspath(os.path.join(__file__, "../../../../"))
    file_path = os.path.join(project_root, "data", "banned_entities.txt")
    banned_entities = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            banned_entities.add(line.strip().lower())
    return banned_entities


def postprocess_named_entities(date, doc, comment_and_score_pairs):
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
    useful_ents = [ent for ent in doc.ents if ent.label_ in ALLOWED_LABELS]
    entity_name_to_label = dict()
    for ent in useful_ents:
        entity_name_to_label[ent.text] = NamedEntityLabel[ent.label_]
    entities = Counter([ent.text for ent in useful_ents])
    banned_entities = load_banned_entities()
    filtered_entities = Counter()
    for name, count in entities.items():
        if not (isinstance(name, str) and filter_named_entity(name)): continue
        if isinstance(name, numbers.Number) or name.isnumeric(): continue
        if len(name) == 1: continue
        if count == 1: continue 
        if name.lower() in banned_entities: continue 
        filtered_entities[name] = count

    # Get top entities and analyze
    #top_entities = filtered_entities.most_common(TIME_FILTER_TO_NAMED_ENTITIES_LIMIT[config['time_filter']])
    top_entities = [(entity_name, count) for entity_name, count in filtered_entities.items()]
    top_entity_names = set([entity[0] for entity in top_entities])
    print('len(top_entity_names): ', len(top_entity_names))
    print('top_entity_names: ', top_entity_names)
    t1 = time.time()
    entity_to_key_points, entity_to_comments = get_key_points_of_top_entities(top_entity_names, doc, comment_and_score_pairs, entity_name_to_label)
    t2 = time.time()
    print('Computing key points of named entities in ', date, ' took: ', t2 - t1)

    for i in range(len(top_entities)):
        entity = NamedEntity(
            name = top_entities[i][0],
            label = entity_name_to_label[top_entities[i][0]].name, # mongodb doesn't let you store enums
            count = top_entities[i][1]
        )
        if entity.name in entity_to_key_points and entity.name in entity_to_comments:
            entity.key_points = entity_to_key_points[entity.name]
            entity.num_comments_summarized = len(entity_to_comments[entity.name])
            top_entities[i] = entity 
        else:
            top_entities[i] = None
    top_entities = [top_entity for top_entity in top_entities if top_entity is not None]
    
    # make possessive entities (ends in 's or s') non-possessive
    for i in range(len(top_entities)):
        if top_entities[i].name.endswith(("'s", "s'", "’s", "s’")):
            top_entities[i].name = top_entities[i].name[:-2]

    return top_entities 


def is_meaningful_mention(sentence, entity_name):
    entity_name_words = entity_name.split()
    doc = dep_parsing_nlp(sentence)
    for token in doc:
        if entity_name_words[-1] in token.text:
            print("token.dep_: ", token.dep_)
            if token.dep_ in {"nsubj", "nsubjpass", "dobj", "agent"}:
                return True
    return False

def get_relevant_parts_of_full_comment(entity_name, sent_text, full_comment):
    '''
        we don't want to use the full_comment if for example, the comment is super long
        and the entity is only mentioned once in the comment. This will confuse the 
        summarizer and the absa model. 
        
        split full_comment up into its sentences 
        search left to the sentence and right to sentence 
        stop searching left once we reach a sentence that doesn't mention the entity OR its the last sentence
        stop searching right once we reach a sentence that doesn't mention the entity OR its the first sentence 
        
        Example: 
        ENTITY = "Trump", SENTENCE = "Trump recently imposed tariffs"
        I've spoken at length about this before. But here's my thoughts again. Trump has orange hair. 
        Trump recently imposed tariffs. These new tariffs were pretty controversial. I personally don't 
        agree with them 
        
        the sentences we would keep in the comment are:
        But here's my thoughts again. Trump has orange hair. 
        Trump recently imposed tariffs. These new tariffs were pretty controversial. I personally don't 
        agree with them 
    '''
    full_comment_sentences_keeping = []
    full_comment_sentences = nltk.sent_tokenize(full_comment)
    for x in range(len(full_comment_sentences)):
        sentence = full_comment_sentences[x]
        if sent_text.lower().strip() in sentence.lower().strip() or sentence.lower().strip() in sent_text.lower().strip():
            full_comment_sentences_keeping.append(sent_text)
            # search left 
            if x > 0:
                # range (x, y, z) = start of range (inclusive), end of range (exclusive), incremental value
                for i in range(x - 1, -1, -1):
                    full_comment_sentences_keeping.insert(0, full_comment_sentences[i])
                    if entity_name.lower() not in full_comment_sentences[i]:
                        break

            # search right 
            if x < len(full_comment_sentences) - 1:
                for i in range(x + 1, len(full_comment_sentences)):
                    full_comment_sentences_keeping.append(full_comment_sentences[i])
                    if entity_name.lower() not in full_comment_sentences[i]:
                        break
            print(f"kept {len(full_comment_sentences_keeping)}/{len(full_comment_sentences)} sentences in full_comment!")
            break
        
    if len(full_comment_sentences_keeping) == 0:
        print("hmm thats weird... couldn't find sent_text in sentence tokenized full_comment")
        return full_comment 
    return " ".join(full_comment_sentences_keeping)


def get_key_points_of_top_entities(top_entity_names, doc, comment_and_score_pairs, entity_name_to_label):
    """Get sentiment and key points for top entities from a spaCy document.
    
    Args:
        top_entity_names: List of entity names to analyze
        doc: spaCy Doc object
        comment_and_score_pairs: Dictionary mapping a comment's text to its score 
    Returns:
        entity_to_sentiment: Dictionary mapping entity name to its sentiment score
        entity_to_key_points: Dictionary mapping entity name to its key points
    """
    entity_to_comments = dict() # Dictionary mapping entity name to a set of the comments its mentioned in
    entity_to_key_points = dict() # Dictionary mapping entity name to a list of its computed key points 
    for ent in doc.ents:
        ent_text = ent.text 
        if ent_text in top_entity_names:
            sent_text = ent.sent.text.strip() 
            if not is_meaningful_mention(sent_text, ent_text):
                print(f"discarded {ent_text} in the sentence: {sent_text}")
                continue
            else:
                print(f"kept {ent_text} in the sentence: {sent_text}")
            full_comment = None
            parsed_full_comment = None 
            for comment_text in comment_and_score_pairs.keys():
                if sent_text.lower() in comment_text.lower():
                    full_comment = comment_text 
                    parsed_full_comment = get_relevant_parts_of_full_comment(ent_text, sent_text, full_comment)
                    break
            if parsed_full_comment is not None:
                print('found parsed_full_comment')
                if ent_text not in entity_to_comments: 
                    entity_to_comments[ent_text] = set()
                if (parsed_full_comment, comment_and_score_pairs[full_comment]) not in entity_to_comments[ent_text]:
                    entity_to_comments[ent_text].add((parsed_full_comment, comment_and_score_pairs[full_comment]))
            else:
                print('could NOT find parsed_full_comment')
    
    # Only keep entities which are mentioned in atleast 3 comments 
    entity_to_comments = {entity: comments for entity, comments in entity_to_comments.items() if len(comments) >= 3}
    # Combine entities that refer to the same thing 
    entity_to_comments = combine_same_entities(entity_to_comments)

    for entity, comments in entity_to_comments.items():
        # comments is a list of tuples (comment, score)
        comments = sorted(comments, key=lambda comment: comment[1])
        max_num_comments = TIME_FILTER_TO_NUM_COMMENTS_PER_ENTITY_LIMIT[config['time_filter']]
        comments = comments[-max_num_comments:] 
        entity_to_comments[entity] = [comment[0] for comment in comments]
        print('kept ', len(comments), ' comments for ', entity)
    
    for entity in entity_to_comments:
        entity_to_key_points[entity] = summarize_comments(entity, config['subreddit'], entity_to_comments[entity])
        time.sleep(5)  # Wait for 5 seconds so we don't exceed Groq's free api rate limit 
    return entity_to_key_points, entity_to_comments


def print_entity_to_txt_file(entity_name, clustered_comments, key_points):
    # Create and write to a file named [entity_name].txt
    with open(f"{entity_name}.txt", "w") as file:
        file.write(f"ENTITY NAME: {entity_name}\n\n")
        file.write("KEY POINTS: \n")
        for key_point, sentiment_score in key_points:
            file.write(f"- sentiment score = {sentiment_score} \n {key_point}\n")
        file.write("-" * 40 + "\n")
        file.write("COMMENTS: \n")
        for cluster_label, comments in clustered_comments.items():
            file.write(f"cluster_label: {cluster_label}\n")
            for x in range(len(comments)):
                file.write(f"{x}.) {comments[x]}\n")
            file.write("\n")


def load_synonym_map():
    project_root = os.path.abspath(os.path.join(__file__, "../../../../"))
    file_path = os.path.join(project_root, "data", "synonyms.txt")
    synonym_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            # Split the line into synonyms
            synonyms = [s.strip().lower() for s in line.strip().split('-')]
            for synonym in synonyms:
                if synonym:
                    synonym_map[synonym] = line_number
    return synonym_map


def combine_same_entities(entity_to_comments):
    # case insensitive combining 
    entities = list(entity_to_comments.keys())
    entities_lowercase = [entity.lower() for entity in entities]
    synonym_map = load_synonym_map()
    for i, x in combinations(range(len(entities_lowercase)), 2):
        entity_1 = entities_lowercase[i]
        entity_2 = entities_lowercase[x]
        entities_are_equal_conditions = [
            # the entities are the same if we make them both plural 
            entity_1 == (entity_2 + "s") or (entity_1 + "s") == entity_2,
            # the entities are the same if we make them both possessive plural 
            entity_1 == (entity_2 + "s'") or (entity_1 + "s'") == entity_2,
            entity_1 == (entity_2 + "s’") or (entity_1 + "s’") == entity_2,
            # the entities are the same if we make them both possessive singular 
            entity_1 == (entity_2 + "'s") or (entity_1 + "'s") == entity_2,
            entity_1 == (entity_2 + "s’") or (entity_1 + "s’") == entity_2,
            entity_1 == (entity_2 + "'d") or (entity_1 + "'d") == entity_2,
            entity_1 == (entity_2 + "’d") or (entity_1 + "’d") == entity_2,
            entity_1 == (entity_2 + "'ve") or (entity_1 + "'ve") == entity_2,
            entity_1 == (entity_2 + "’ve") or (entity_1 + "’ve") == entity_2,
            entity_1 == (entity_2 + "'re") or (entity_1 + "'re") == entity_2,
            entity_1 == (entity_2 + "’re") or (entity_1 + "’re") == entity_2,
            entity_1 == (entity_2 + "n't") or (entity_1 + "n't") == entity_2,
            entity_1 == (entity_2 + "n’t") or (entity_1 + "n’t") == entity_2,
            # the entities are the same if we make them both one word 
            ''.join(entity_1.split()) == ''.join(entity_2.split()),
            # the 2nd entity is a letter abbreviation of the 1st entity 
            (len(entity_1.split()) >= 2 and ''.join([word[0] for word in entity_1.split()]) == entity_2),
            # the 1st entity is a letter abbreviation of the 2nd entity 
            (len(entity_2.split()) >= 2 and ''.join([word[0] for word in entity_2.split()]) == entity_1),
            # the 2nd entity is just the first or last name of the full name 1st entity 
            (len(entity_1.split()) == 2 and len(entity_2.split()) == 1 and 
                (entity_1.split()[0] == entity_2 or entity_1.split()[1] == entity_2)),
            # the 1st entity is just the first or last name of the full name 2nd entity 
            (len(entity_2.split()) == 2 and len(entity_1.split()) == 1 and 
                (entity_2.split()[0] == entity_1 or entity_2.split()[1] == entity_1)),
            # the entities are both in the synonym map and are on the same line # 
            entity_1 in synonym_map and entity_2 in synonym_map and 
            synonym_map[entity_1] == synonym_map[entity_2]
            
        ]
        if any(entities_are_equal_conditions):
            print("At least one equal condition is true! combining ", entities[i], " and ", entities[x])
            entity_to_comments = combine_two_entities(entity_to_comments, entities[i], entities[x])

    return entity_to_comments 


def combine_two_entities(entity_to_comments, entity_1, entity_2):
    if entity_1 is None or entity_2 is None or entity_1 not in entity_to_comments or entity_2 not in entity_to_comments:
        return entity_to_comments
    
    if len(entity_1) >= len(entity_2): longer, shorter = entity_1, entity_2
    else: longer, shorter = entity_2, entity_1

    if (longer == shorter + "'s" or longer == shorter + "s'" or 
        longer == shorter + "’s" or longer == shorter + "s’" or 
        longer == shorter + "'d" or longer == shorter + "’d" or 
        longer == shorter + "'ve" or longer == shorter + "’ve" or
        longer == shorter + "'re" or longer == shorter + "’re" or
        longer == shorter + "n't" or longer == shorter + "n’t"): 
        entity_to_comments[shorter] = entity_to_comments[shorter] | entity_to_comments[longer]
        del entity_to_comments[longer]
    else:
        entity_to_comments[longer] = entity_to_comments[longer] | entity_to_comments[shorter]
        del entity_to_comments[shorter]
    return entity_to_comments 


def get_flattened_text_for_comment(comment, indent):
    return (" " * indent) + comment.text + "\n" + "".join([get_flattened_text_for_comment(reply, indent + 4) for reply in comment.replies])


# Assuming coreference resolution has already been performed when this function gets called 
def get_flattened_text_for_post(post):
    flattened_text = "Title: " + post.title + "\n" + "Description: " + post.description 
    for top_level_comment in post.top_level_comments:
        flattened_text += " " + get_flattened_text_for_comment(top_level_comment, indent = 0)
    return flattened_text 

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
                post_content = get_flattened_text_for_post(post)
                if entity.name in post_content: 
                    try:
                        post_url = "https://www.reddit.com" + post.permalink
                        posts_that_mention_entity.append((post_url, post_content.count(entity.name), post.score))
                    except Exception as e:
                        print('couldnt get the post permalink because: ', e)
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
