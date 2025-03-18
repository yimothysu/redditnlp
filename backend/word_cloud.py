from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json

wc = WordCloud(background_color="white", max_words=2000, contour_width=3, contour_color='steelblue')

# Takes in dictionary of named entities
def generate_word_cloud(named_entities):    
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

# Takes in dictionary of named entities
def generate_by_date(named_entities):
    # Add named entities to dictionary
    for date, entity_list in named_entities.items():
        
        tempDict = {}        
        for entity in entity_list:
            if entity[0] not in tempDict:
                tempDict[entity[0]] = entity[1]
            elif entity[0] in tempDict:
                tempDict[entity[0]] += entity[1]
        
        # If no named entities for date, skip
        if not entity_list:
            continue
                
        wc.generate_from_frequencies(tempDict)
        print("Word cloud generated")
        
        return wc.to_array()
