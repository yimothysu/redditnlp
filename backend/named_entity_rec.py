import spacy 
import requests 
import numbers
from main import RedditPost 
from collections import Counter

nlp = spacy.load("en_core_web_sm")

url = "http://localhost:8000/sample/popculturechat"
response = requests.get(url)
data = response.json()
restored_dict = {
    key: [RedditPost.model_validate(post) for post in posts]  # Convert each post to RedditPost
    for key, posts in data.items()
}

# idea: we can calculate sentiment towards these named entities !!! 
# and have a 'common words users used to describe this named entity' 
for date, posts in restored_dict.items():
    post_content = ' '.join([post.title + ' '.join(post.comments) for post in posts])
    doc = nlp(post_content)
    
    # named_entities = [(ent.text, ent.label_) for ent in doc.ents]
    named_entities = Counter([ent.text for ent in doc.ents])
    banned_entities = ['one', 'two', 'first', 'second', 'yesterday', 'today', 'tomorrow', 
                       'approx', 'half', 'idk']
    filtered_named_entities = Counter()
    for name, count in named_entities.items():
        if isinstance(name, numbers.Number) or name.isnumeric() or name.lower().strip() in banned_entities: continue
        filtered_named_entities[name] = count
    
    filtered_top_named_entities = filtered_named_entities.most_common(10)
    print('key: ', date)
    print('top_named_entities: ')
    for x in range(10):
        print(x + 1, '.) ', filtered_top_named_entities[x][0])
    print('\n')