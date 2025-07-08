"""Data models for subreddit analysis and queries.
Defines Pydantic models for handling subreddit data, analysis results, and metrics.
"""
from __future__ import annotations
from pydantic import BaseModel
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any

class SubredditQuery(BaseModel):
    """Parameters for querying subreddit data."""
    name: str # Name of the subreddit
    time_filter: str # Time period to analyze ('week', 'year', 'all')

    def to_composite_key(self) -> str:
        """Generate a unique key for the query."""
        return f"{self.name}#{self.time_filter}"


class NamedEntityLabel(Enum):
    PERSON = 0 # Named people (e.g., Elon Musk, Marie Curie)
    ORG = 1 # Organizations (e.g., Google, UN, Harvard University)
    GPE = 2 # Geopolitical Entities: countries, cities, states (e.g., France, New York)
    LOC = 3 #  Non-GPE locations (e.g., Pacific Ocean, Mount Everest)
    FAC = 4 # Facilities (e.g., Golden Gate Bridge, Statue of Liberty)
    PRODUCT = 5 # Tangible products (e.g., iPhone, Tesla Model S)
    WORK_OF_ART = 6 # Titles of creative works (e.g., The Mona Lisa, 1984)
    LAW = 7 # Named documents or legal rules (e.g., Constitution, GDPR)
    EVENT = 8 # Named events (e.g., World War II, Super Bowl)
    LANGUAGE = 9 # Named languages (e.g., French, Mandarin)
    NORP = 10 # Nationalities, religious and political groups (e.g., American, Buddhist, Democrat)
 

class NamedEntity(BaseModel):
    """Named entity extracted from subreddit text with associated metrics."""
    name: str # Name of the entity 
    label: Optional[str] = None # the NamedEntityLabel of the entity (converted to string because mongodb doesn't support enums)
    key_points: Optional[str] = None # Key discussion points about the entity 
    num_comments_summarized: Optional[int] = None # How many comments synthesized into key_points 
    urls: Optional[List[str]] = None # Related URLs


class NGram(BaseModel):
    """N-gram frequency data."""
    name: str # The n-gram text
    count: int # Number of times the n-gram appears in the text 


class Comment(BaseModel):
    """Data model for a Reddit comment.
    
    Attributes:
        text: The comment's content
        score: The comment's score (upvotes - downvotes)
        replies: The comment's replies 
    """
    text: Optional[str] = None 
    # the context needed (title & description for top level comments) to resolve the pronouns in the comment 
    context: Optional[str] = None 
    score: Optional[int] = None 
    replies: Optional[List[Comment]] = None 


class RedditPost(BaseModel):
    """Data model for a Reddit post with its metadata and comments.
    
    Attributes:
        title: Post title
        description: Post content/selftext
        score: Post score (upvotes - downvotes)
        url: URL of the post
        created_utc: Post creation timestamp in UTC
        num_comments: Total number of comments
        top_level_comments: List of top level comment texts
        comment_scores: List of top level comment scores
    """
    title: Optional[str] = ""
    description: Optional[str] = "" 
    idx: Optional[int] = -1 
    url: Optional[str] = "" 
    score: Optional[int] = None
    created_utc: Optional[float] = None
    num_comments: Optional[int] = None
    top_level_comments: Optional[list[Comment]] = None 
    
    def to_dict(self):
        """Convert the post data to a dictionary format."""
        return self.model_dump()
    

class SubredditAnalysis(BaseModel):
    timestamp: int # when the analysis was generated 
    num_words: int # Total word count analyzed 
    subreddit: str # Name of the subreddit 
    #  key = date
    top_n_grams: Optional[Dict[str, List[NGram]]] = None # Top n-grams by date 
    # key = topic, value = list of post links relating to topic 
    topics: Optional[Dict[str, list[RedditPost]]] = None 
    # key = date 
    top_named_entities: Optional[Dict[str, List[NamedEntity]]] = None # Top named entities by date 
    top_named_entities_embeddings: Optional[Dict[str, Tuple[float, float]]] = None # Embeddings of the top named entities by date 
    top_named_entities_wordcloud: Optional[str] = None # Wordcloud of the top named entities by date 
    readability_metrics: Optional[Dict[str, Any]] = None # Readability metrics by date 

    # For subreddit ranking
    toxicity_score: Optional[float] = None # [0, 1] --> 0 = not toxic at all, 1 = all toxic 
    toxicity_grade: Optional[str] = None # A+ to F 
    toxicity_percentile: Optional[float] = None # [0, 100]
    all_toxicity_scores: Optional[List[float]] = None # for generating a toxicity scores distribution graph on the UI 
    all_toxicity_grades: Optional[List[str]] = None # for generating a toxicity grades distribution graph on the UI 

    positive_content_score: Optional[float] = None # [0, 1] --> 0 = no positive content, 1 = all positive content
    positive_content_grade: Optional[str] = None # A+ to F 
    positive_content_percentile: Optional[float] = None # [0, 100]
    all_positive_content_scores: Optional[List[float]] = None # for generating a positive content scores distribution graph on the UI 
    all_positive_content_grades: Optional[List[str]] = None # for generating a positive content grades distribution graph on the UI
