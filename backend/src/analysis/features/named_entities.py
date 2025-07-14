"""Module for extracting and analyzing named entities from subreddit content.

Provides functionality for:
- Named entity recognition using spaCy
- Sentiment analysis using ABSA (Aspect-Based Sentiment Analysis)
- Key point extraction using BART summarization
- Entity consolidation and filtering
"""

import time, re, nltk, numbers, os  # type: ignore
import spacy  # type: ignore
from src.utils.subreddit_classes import ( NamedEntity, NamedEntityLabel, RedditPost )
from src.analysis.groq import ( summarize_comments )
from itertools import combinations

# Configuration constants
TIME_FILTER_TO_NAMED_ENTITIES_LIMIT = {'all': 75, 'year': 50, 'week': 25}
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

def get_flattened_text_for_comment(comment, indent):
    indent_str = " " * indent
    indented_lines = [(indent_str + line) for line in comment.text.splitlines()]
    result = "\n".join(indented_lines) + "\n"
    for reply in comment.replies:
        result += get_flattened_text_for_comment(reply, indent + 4)
    return result


# Assuming coreference resolution has already been performed when this function gets called 
def get_flattened_text_for_post(post, include_comments=False):
    flattened_text = "Title: {title}\nDescription: {description}".format(title = post.title, description = post.description)
    if include_comments: 
        flattened_text += "Comments: \n"
        for top_level_comment in post.top_level_comments:
            flattened_text += " " + get_flattened_text_for_comment(top_level_comment, indent = 0)
    return flattened_text 

def find_comment_with_sentence(sent):
    if sent is None:
        return ""
    return sent 

def get_top_entities(subreddit: str, time_filter: str, posts: list[RedditPost]):
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
    post_content = '\n'.join([get_flattened_text_for_post(post, include_comments=True) for post in posts])
    # split post_content into 950,000 character chunks so we don't exceed the model's limit of 1,000,000 characters per text 
    post_content_chunks = [post_content[i:i+950000] for i in range(0, len(post_content), 950000)]
    all_named_entities = dict() # key = (name, label), value = NamedEntity object 
    for chunk in post_content_chunks:
        doc = nlp(chunk)
        for ent in doc.ents:
            if (ent.text, ent.label_) not in all_named_entities:
                # TO DO: change ent.sent to the actual comment with the context 
                all_named_entities[(ent.text, ent.label_)] = NamedEntity(name=ent.text, freq=1, label=ent.label_, comments=[find_comment_with_sentence(ent.sent.text)])
            else:
                named_entity = all_named_entities[(ent.text, ent.label_)]
                named_entity.comments.append(find_comment_with_sentence(ent.sent))
                named_entity.freq += 1 
                all_named_entities[(ent.text, ent.label_)] = named_entity 
    # get rid of comment duplicates for each named entity 
    all_named_entities = list(all_named_entities.values()) 
    for i in range(len(all_named_entities)):
        entity = all_named_entities[i]
        comments_list = entity.comments 
        comments_list_unique = list(set(comments_list))
        entity.comments = comments_list_unique
        all_named_entities[i] = entity 
    all_named_entities = postprocess_named_entities(all_named_entities)
    return all_named_entities


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


def postprocess_named_entities(all_named_entities: list[NamedEntity]):
    banned_entities = load_banned_entities()
    filtered_entities = []
    for ent in all_named_entities:
        if ent.label not in ALLOWED_LABELS: continue 
        if not (isinstance(ent.name, str) and filter_named_entity(ent.name)): continue
        if isinstance(ent.name, numbers.Number) or ent.name.isnumeric(): continue
        if len(ent.name) == 1: continue
        if ent.freq == 1: continue 
        if ent.name.lower() in banned_entities: continue 
        if ent is None: continue 
        if ent.name.endswith(("'s", "s'", "’s", "s’")): ent.name = ent.name[:-2]
        filtered_entities.append(ent)
    filtered_entities = combine_same_entities(filtered_entities)
    # sort the entities by their freq attribute and select the top occurring named entities 
    filtered_entities.sort(key=lambda ent: ent.freq)
    filtered_entities = filtered_entities[-TIME_FILTER_TO_NAMED_ENTITIES_LIMIT[config['time_filter']]:]
    filtered_entities = get_AI_analysis_of_top_entities(filtered_entities)
    # empty the comments attribute for each entity to save mongodb space 
    for i in range(len(filtered_entities)):
        entity = filtered_entities[i]
        entity.comments = []
        filtered_entities[i] = entity
    return filtered_entities 


def get_AI_analysis_of_top_entities(top_entities: list[NamedEntity]):
    for i in range(len(top_entities)):
        entity = top_entities[i]
        entity.AI_analysis = summarize_comments(entity.name, config['subreddit'], entity.comments)
        top_entities[i] = entity 
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


def combine_same_entities(named_entities: list[NamedEntity]):
    # case insensitive combining 
    entities_lowercase = [entity.name.lower() for entity in named_entities]
    synonym_map = load_synonym_map()
    combined_named_entities = named_entities 
    for i, x in combinations(range(len(entities_lowercase)), 2):
        entity_1 = entities_lowercase[i]
        entity_2 = entities_lowercase[x]
        entities_are_equal_conditions = [
            entity_1 == entity_2,
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
            entity_1 == (entity_2 + "'ll") or (entity_1 + "'ll") == entity_2,
            entity_1 == (entity_2 + "’ll") or (entity_1 + "’ll") == entity_2,
            entity_1 == ("the " + entity_2) or ("the " + entity_1) == entity_2,
            entity_1 == ("a " + entity_2) or ("a " + entity_1) == entity_2,
            entity_1 == ("an " + entity_2) or ("an " + entity_1) == entity_2,
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
            print("At least one equal condition is true! combining ", entity_1, " and ", entity_2)
            combined_entity = combine_two_entities(named_entities[i], named_entities[x])
            # delete named_entities[i] and named_entities[x] from combined_named_entities, and then add combined_entity
            combined_named_entities = [e for e in combined_named_entities if e != named_entities[i]]
            combined_named_entities = [e for e in combined_named_entities if e != named_entities[x]]
            if combined_entity not in combined_named_entities:
                combined_named_entities.append(combined_entity)
    return combined_named_entities 



def combine_two_entities(entity_1: NamedEntity, entity_2: NamedEntity):
    if len(entity_1.name) >= len(entity_2.name): longer, shorter = entity_1, entity_2
    else: longer, shorter = entity_2, entity_1

    if (longer.name == shorter.name + "'s" or longer.name == shorter.name + "s'" or 
        longer.name == shorter.name + "’s" or longer.name == shorter.name + "s’" or 
        longer.name == shorter.name + "'d" or longer.name == shorter.name + "’d" or 
        longer.name == shorter.name + "'ve" or longer.name == shorter.name + "’ve" or
        longer.name == shorter.name + "'re" or longer.name == shorter.name + "’re" or
        longer.name == shorter.name + "n't" or longer.name == shorter.name + "n’t" or 
        longer.name == shorter.name + "'ll" or longer.name == shorter.name + "’ll"): 
        # keep the shorter entity 
        combined_entity = shorter 
        combined_comments = set(combined_entity.comments)
        combined_comments.update(longer.comments)
        combined_entity.comments = list(combined_comments)
        return combined_entity 
    else:
        # keep the longer entity 
        combined_entity = longer 
        combined_comments = set(combined_entity.comments)
        combined_comments.update(shorter.comments)
        combined_entity.comments = list(combined_comments)
        return combined_entity 


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
