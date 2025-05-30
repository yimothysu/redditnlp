import re, os, string, spacy, nltk  # type: ignore
import gender_guesser.detector as gender_guesser
from spacy.symbols import ORTH
from collections import namedtuple


nlp = spacy.load("en_core_web_sm")

def load_famous_people_gender():
    project_root = os.path.abspath(os.path.join(__file__, "../../../"))
    file_path = os.path.join(project_root, "data", "famous_people_gender.txt")
    famous_people_gender = dict()
    with open(file_path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            famous_person_name, gender = line.strip().lower().split(" - ")
            famous_people_gender[famous_person_name] = gender 
    return famous_people_gender


def join_contractions_in_tokens(doc, verbose=False):
    if verbose: 
        print("doc before joining contractions: ")
        for token in doc:
            print(f"token.text: {token.text}, token.pos_: {token.pos_}, token.dep_: {token.dep_}")
    
    Token = namedtuple('Token', ['text', 'pos_', 'dep_'])
    joined_tokens = []
    skip_next = False
    for i in range(len(doc)):
        if skip_next:
            skip_next = False
            continue
        if i < len(doc) - 1 and doc[i+1].text in ["'s", "’s", "'d", "’d", "'re", "’re", "'ve", "’ve", "n't", "n’t"]:
            joined_tokens.append(Token(doc[i].text + doc[i+1].text, doc[i].pos_, doc[i].dep_))
            skip_next = True
        else:
            joined_tokens.append(doc[i])

    if verbose: 
        print("doc after joining contractions: ")
        for token in joined_tokens:
            print(f"token.text: {token.text}, token.pos_: {token.pos_}, token.dep_: {token.dep_}")
    
    return joined_tokens

def detect_gender(entity_name):
    # pronouns: she/her, he/him
    famous_people_gender = load_famous_people_gender()
    if entity_name in famous_people_gender:
        gender = famous_people_gender[entity_name]
    else:
        gender_detector = gender_guesser.Detector()
        first_name = entity_name.split(" ")[0]
        first_name = first_name[0].upper() + first_name[1:]
        gender = gender_detector.get_gender(first_name)
        if gender == "mostly_female": 
            gender = "female"
        if gender == "mostly_male":
            gender = "male"
        if gender == "unknown":
            return sentence # don't change the sentence, the person's gender is ambiguous 
    return gender 


'''
    replaces word with she/she's/her/hers 
        if possessive, use her  --> heuristic: the following word is usually a noun 
        if contraction, use she's  --> heuristic: the following word is usually a verb or adverb 
        Use dependency parsing to figure out the grammatical role of the following word 
'''
def replace_female_entity_with_pronoun(word: str, entity_name: str, idx: int, sentence_words: list[str]):
    if (word.endswith('\'d') or word.endswith('’d')) and not (entity_name.endswith('\'d') or entity_name.endswith('’d')):
        return "she'd"
    elif word.endswith('\'s') or word.endswith('’s'):
        doc = nlp(sentence)
        doc = join_contractions_in_tokens(doc)
        if idx + 1 < len(sentence_words):
            next_word_pos = doc[idx + 1].pos_
            if next_word_pos == "NOUN":
                return "her"
            elif next_word_pos == "VERB" or next_word_pos == "ADV":
                return "she's"
            else:
                return word # if next_word_pos == "ADJ", it's unclear whether to use his or he's 
        elif idx == len(sentence_words - 1): # if the word is the last word in the sentence, it's almost definitely possessive 
            return "hers"
    elif word.endswith('s') and not entity_name.endswith('s'):
        return word
    else:
        return "she"

'''
    replace word with he/he's/him/his 
    if possessive, use his --> heuristic: the following word is usually a noun 
    if contraction, use he's --> heuristic: the following word is usually a verb or adjective 
    Use dependency parsing to figure out the grammatical role of the following word 
'''
def replace_male_entity_with_pronoun(word: str, entity_name: str, idx: int, sentence_words: list[str]):
    if (word.endswith('\'d') or word.endswith('’d')) and not (entity_name.endswith('\'d') or entity_name.endswith('’d')):
        return "he'd"
    elif word.endswith('\'s') or word.endswith('’s'):
        doc = nlp(sentence)
        doc = join_contractions_in_tokens(doc)
        if idx + 1 < len(sentence_words):
            next_word_pos = doc[idx + 1].pos_
            if next_word_pos == "NOUN":
                return "his"
            elif next_word_pos == "VERB" or next_word_pos == "ADV":
                return "he's"
            else:
                return word # if next_word_pos == "ADJ", it's unclear whether to use his or he's 
        elif idx == len(sentence_words - 1): # if the word is the last word in the sentence, it's almost definitely possessive 
            return "his"
    elif (word.endswith('s')) and not entity_name.endswith('s'):
        return word 
    else:
        return "he"    


def replace_norp_entity_with_pronoun(word: str, entity_name: str, idx: int, sentence_words: list[str]):
    # replace word with they/them/their/they're 
    if (word.endswith('\'d') or word.endswith('’d')) and not (entity_name.endswith('\'d') or entity_name.endswith('’d')):
        return "they'd"
    elif word.endswith('\'s') or word.endswith('’s'):
        doc = nlp(sentence)
        doc = join_contractions_in_tokens(doc)
        if idx + 1 < len(sentence_words):
            next_word_pos = doc[idx + 1].pos_
            if next_word_pos == "NOUN":
                return "their"
            elif next_word_pos == "VERB" or next_word_pos == "ADV":
                return "they're"
            else:
                return word # if next_word_pos == "ADJ", it's unclear whether to use his or he's 
        elif idx == len(sentence_words - 1): # if the word is the last word in the sentence, it's almost definitely possessive 
            return "their"
    elif (word.endswith('s')) and not entity_name.endswith('s'):
        return word 
    else:
        return "they"    
    return word 


def replace_object_entity_with_pronoun(word: str, entity_name: str, idx: int, sentence_words: list[str]):
    # replace word with it/its/it'd
    if (word.endswith('\'d') or word.endswith('’d')) and not (entity_name.endswith('\'d') or entity_name.endswith('’d')):
        return "it'd"
    elif word.endswith('\'s') or word.endswith('’s'):
        doc = nlp(sentence)
        doc = join_contractions_in_tokens(doc)
        if idx + 1 < len(sentence_words):
            next_word_pos = doc[idx + 1].pos_
            if next_word_pos == "NOUN":
                return "its"
            elif next_word_pos == "VERB" or next_word_pos == "ADV":
                return "it's"
            else:
                return word # if next_word_pos == "ADJ", it's unclear whether to use his or he's 
        elif idx == len(sentence_words - 1): # if the word is the last word in the sentence, it's almost definitely possessive 
            return "its"
    elif (word.endswith('s')) and not entity_name.endswith('s'):
        return word 
    else:
        return "it"    
    return word 


def split_sentence_into_words(sentence: str):
    # remove punctuation before splitting 
    punctuation = ""
    while(sentence[-1] in string.punctuation):
        punctuation += sentence[-1]
        sentence = sentence[:-1]
    sentence_words = sentence.split(" ")
    return punctuation, sentence_words 


def make_sentence_natural_sounding(entity_name: str, entity_type: str, sentence: str):
    # case insensitive 
    sentence = sentence.strip()
    entity_name = entity_name.strip().lower() 
    punctuation, sentence_words = split_sentence_into_words(sentence)
    matches = [m.start() for m in re.finditer(rf"\b{re.escape(entity_name)}\b", sentence)]
    if len(matches) <= 1: return sentence  
    
    new_sentence_words = []
    if entity_type == "PERSON":
        gender = detect_gender(entity_name) 
        first_entity_name_mention = True 
        idx = 0
        while idx < len(sentence_words):
            increment_idx = True 
            word = sentence_words[idx]
            word_case_insensitive = word.lower() 
            if len(entity_name.split(" ")) > 1 and word_case_insensitive.startswith(entity_name.split(" ")[0]):
                offset = 1 
                while idx + offset < len(sentence_words) and offset < len(entity_name.split(" ")) and sentence_words[idx + offset].startswith(entity_name.split(" ")[offset]):
                    word_case_insensitive += " " + sentence_words[idx + offset].strip().lower()
                    word += " " + sentence_words[idx + offset].strip() 
                    offset += 1 
                idx = idx + offset  
                increment_idx = False

            if (word_case_insensitive.startswith(entity_name) or entity_name.startswith(word_case_insensitive)) and first_entity_name_mention:
                new_sentence_words.append(word)
                first_entity_name_mention = False
                idx += 1 
                continue 
            elif (word_case_insensitive.startswith(entity_name) or entity_name.startswith(word_case_insensitive)) and not first_entity_name_mention:
                if gender == "female":
                    new_sentence_words.append(replace_female_entity_with_pronoun(word, entity_name, idx, sentence_words))
                elif gender == "male":
                    new_sentence_words.append(replace_male_entity_with_pronoun(word, entity_name, idx, sentence_words))
            else:
                new_sentence_words.append(word)
            if increment_idx: idx += 1 
    elif entity_type == "NORP":
        # pronouns: they/them/theirs/their 
        first_entity_name_mention = True 
        for word in sentence_words: 
            word_case_insensitive = word.lower() 
            if word_case_insensitive.startswith(entity_name) and first_entity_name_mention:
                new_sentence_words.append(word)
                first_entity_name_mention = False
                continue 
            elif word_case_insensitive.startswith(entity_name) and not first_entity_name_mention:
                new_sentence_words.append(replace_norp_entity_with_pronoun(word, entity_name, idx, sentence_words))
    elif entity_type in  {"ORG", "GPE", "LOC", "FAC", "PRODUCT", "WORK_OF_ART", "LAW", "LANGUAGE"}:
        # pronouns: it/its 
        first_entity_name_mention = True 
        for word in sentence_words: 
            word_case_insensitive = word.lower() 
            if word_case_insensitive.startswith(entity_name) and first_entity_name_mention:
                new_sentence_words.append(word)
                first_entity_name_mention = False
                continue 
            elif word_case_insensitive.startswith(entity_name) and not first_entity_name_mention:
                new_sentence_words.append(replace_object_entity_with_pronoun(word, entity_name, idx, sentence_words))
    return " ".join(new_sentence_words) + punctuation


def make_text_natural_sounding(entity_name: str, entity_type: str, text: str):
    sentences = nltk.sent_tokenize(text)
    sentences = [sentence for sentence in sentences if sentence and len(sentence) > 5]
    new_sentences = []
    for sentence in sentences: 
        new_sentences.append(make_sentence_natural_sounding(entity_name, entity_type, sentence))
    return " ".join(new_sentences)


# =========== EXAMPLE USAGE =============
sentence = "Trump said Trump would lead the rally and that Trump’s ideas were Trump’s best."
# Expected output: Trump said they would lead the rally and that their ideas were their best 
print("natural sounding sentence: ", make_text_natural_sounding("trump", "PERSON", sentence), "\n")
sentence = "Grace's running for class president this year, and I think Grace has a good chance!"
# Expected output: Grace's running for class president this year, and I think she has a good chance! 
print("natural sounding sentence: ", make_text_natural_sounding("grace", "PERSON", sentence), "\n")
sentence = "I know people who voted for Trump the first time because they genuinely thought that if nothing else, Trump would shake things up and actually lead to something changing in DC."
print("natural sounding sentence: ", make_text_natural_sounding("Trump", "PERSON", sentence), "\n")
sentence = "Raymond Joseph Teller and Penn Jillette. Teller legally changed his name to the mononym \"Teller\", his family surname."
print("natural sounding sentence: ", make_text_natural_sounding("Teller", "PERSON", sentence), "\n")
sentence = "You could even prop Norman up somewhere for a picture and Norman'd just chill there."
print("natural sounding sentence: ", make_text_natural_sounding("Norman", "PERSON", sentence), "\n")
sentence = "I hope Andrew Tate just leaves everyone alone, regardless of who Andrew Tate's attracted to."
print("natural sounding sentence: ", make_text_natural_sounding("Andrew Tate", "PERSON", sentence), "\n")