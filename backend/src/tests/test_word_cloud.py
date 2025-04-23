import pytest
import sys
import os 
from wordcloud import WordCloud
from PIL import Image
from io import BytesIO
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.features.word_cloud import *

# Test edge cases for the named_entities_to_dictionary function
class TestEdgeCases:
    
    # Test empty input
    def test_empty_named_entities(self):
        input = {}
        expected_output = {}
        
        output = named_entities_to_dictionary(input)
        assert output == expected_output
    
    # Test input with one named entity
    def test_single_entity(self):
        input = {
            "04-20": [("Entity A", 5)],
        }
        
        expected_output = {
            "Entity A": 5,
        }
        
        output = named_entities_to_dictionary(input)
        assert output == expected_output
    
    # Test input with multiple named entities on the different date
    def test_single_entity_multiple_dates(self):
        input = {
            "04-20": [("Entity A", 5)],
            "04-21": [("Entity A", 3)],
        }
        
        expected_output = {
            "Entity A": 8,
        }
        
        output = named_entities_to_dictionary(input)
        assert output == expected_output
    
    # Test input with empty date    
    def test_empty_date(self):
        input = {
            "04-20": [("Entity A", 3)],
            "04-21": [],
        }
        
        expected_output = {
            "Entity A": 3,
        }
        
        output = named_entities_to_dictionary(input)
        assert output == expected_output

# Test the named_entities_to_dictionary function with a simple case
class TestNamedEntitiesToDictionary:
    
    # Test input with multiple named entities
    def test_simple_named_entities_to_dictionary(self):
        input = {
            "04-20": [("Entity A", 5), ("Entity B", 3)],
            "04-21": [("Entity C", 2), ("Entity D", 4)],
            "04-22": [("Entity E", 1), ("Entity F", 6)],
            "04-23": [("Entity G", 2), ("Entity H", 3)],
        }
        
        expected_output = {
            "Entity A": 5,
            "Entity B": 3,
            "Entity C": 2,
            "Entity D": 4,
            "Entity E": 1,
            "Entity F": 6,
            "Entity G": 2,
            "Entity H": 3,
        }
        
        output = named_entities_to_dictionary(input)
        assert output == expected_output

# Test the generate_word_cloud function and the entities_to_dictionary function
class TestGenerateWordCloud:
    wc = WordCloud(background_color="white", max_words=2000, contour_width=3, contour_color='steelblue')

    # Test input with multiple named entities and ensure the word cloud is generated correctly
    def test_valid_word_cloud(self):
        input = {
            "04-20": [("Entity A", 5), ("Entity B", 3)],
            "04-21": [("Entity C", 2), ("Entity D", 4)],
            "04-22": [("Entity E", 1), ("Entity F", 6)],
            "04-23": [("Entity G", 2), ("Entity H", 3)],
        }
        
        expected_frequencies = {
            "Entity A": 5,
            "Entity B": 3,
            "Entity C": 2,
            "Entity D": 4,
            "Entity E": 1,
            "Entity F": 6,
            "Entity G": 2,
            "Entity H": 3,
        }
        
        # Ensure frequencies match before generating the word cloud
        frequencies = named_entities_to_dictionary(input)
        assert frequencies == expected_frequencies
        
        # Check if the word cloud is generated correctly by checking format
        try:
            img_data = base64.b64decode(generate_word_cloud(input))
            img = Image.open(BytesIO(img_data))
            img.verify()
            assert img.format == "PNG"
        except Exception as e:
            assert False, f"Word cloud generation failed: {e}"
            
