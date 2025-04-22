"""Module for generating word clouds from named entities in subreddit content."""

from wordcloud import WordCloud # type: ignore
from io import BytesIO
import base64

def generate_word_cloud(named_entities):    
    """Generate a base64 encoded word cloud image from named entities.
    
    Args:
        named_entities: Dictionary mapping dates to lists of named entities
    Returns:
        str: Base64 encoded PNG image of the word cloud
    """
    # Create dictionary of named entities and their frequencies
    entities_to_freq = {}
    for entity_list in named_entities.values():        
        for entity in entity_list:
            if entity.name not in entities_to_freq:
                entities_to_freq[entity.name] = entity.count
            elif entity.name in entities_to_freq:
                entities_to_freq[entity.name] += entity.count
    
    wc = WordCloud(background_color="white", max_words=2000, contour_width=3, contour_color='steelblue')
    wc.generate_from_frequencies(entities_to_freq)
    buffer = BytesIO()
    wc.to_image().save(buffer, "PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    
    return img_base64

def generate_by_date(named_entities):
    """Generate word cloud arrays for each date's named entities.
    
    Args:
        named_entities: Dictionary mapping dates to lists of named entities
    Returns:
        numpy.ndarray: Word cloud image array for the first non-empty date,
                      or None if no entities found
    """
    # Add named entities to dictionary
    for date, entity_list in named_entities.items():
        tempDict = {}        
        for entity in entity_list:
            if entity.name not in tempDict:
                tempDict[entity.name] = entity.count
            elif entity.name in tempDict:
                tempDict[entity.name] += entity.count
        
        if not entity_list: continue
        
        wc = WordCloud(background_color="white", max_words=2000, contour_width=3, contour_color='steelblue')  
        wc.generate_from_frequencies(tempDict)
        return wc.to_array()
