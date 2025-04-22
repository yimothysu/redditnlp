from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Takes in dictionary of named entities
def generate_word_cloud(named_entities):    
    # Create dictionary of named entities and their frequencies
    fullDict = named_entities_to_dictionary(named_entities)
    
    wc = WordCloud(background_color="white", max_words=2000, contour_width=3, contour_color='steelblue')
    
    wc.generate_from_frequencies(fullDict)
        
    buffer = BytesIO()
    wc.to_image().save(buffer, "PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    
    return img_base64

def named_entities_to_dictionary(named_entities):
    # Create dictionary of named entities and their frequencies
    fullDict = {}
    
    # Add named entities to dictionary
    for entity_list in named_entities.values():        
        for entity in entity_list:
            if entity[0] not in fullDict:
                fullDict[entity[0]] = entity[1]
            elif entity[0] in fullDict:
                fullDict[entity[0]] += entity[1]
    
    return fullDict
    
