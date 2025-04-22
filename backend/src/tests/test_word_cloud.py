import pytest
import sys

sys.path.append("../analysis/features")
from word_cloud import named_entities_to_dictionary

class TestEdgeCases:
    def test_empty_named_entities(self):
        input = {}
        expected_output = {}
        
        output = named_entities_to_dictionary(input)
        assert output == expected_output
        
    def test_single_entity(self):
        input = {
            "04-20": [("Entity A", 5)],
        }
        
        expected_output = {
            "Entity A": 5,
        }
        
        output = named_entities_to_dictionary(input)
        assert output == expected_output
        
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
        
class TestNamedEntitiesToDictionary:
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
    
    