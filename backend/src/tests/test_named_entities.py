import pytest
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.features.named_entities import *

async def test_named_entities():
    post_content_grouped_by_date = {'4-11': 'Google is so cool. I went to California last week. '
                                            'I visited the Google headquarters. Did you hear about the new Google update? ', 
                                    '4-12': 'Nintendo consoles have gone down in recent years. The new Nintendo console was pretty '
                                            'expensive ngl. I dont particularly like or dislike Nintendo - I just dont think their '
                                            'games are that wow'}
    top_entities = await get_top_entities('week', post_content_grouped_by_date, {})
    assert len(top_entities) == 2 
    assert top_entities.keys() == {'4-11', '4-12'}
    for date, entities in top_entities.items():
        if date == '4-11':
            assert len(entities) == 1
            assert entities[0].name == "Google"
            assert entities[0].sentiment > 0
            assert entities[0].count == 3
            assert len(entities[0].key_points) > 0
        if date == '4-12':
            assert len(entities) == 1
            assert entities[0].name == "Nintendo"
            assert entities[0].sentiment < 0
            assert entities[0].count == 3
            assert len(entities[0].key_points) > 0

