"""Data models for subreddit analysis and queries.
Defines Pydantic models for handling subreddit data, analysis results, and metrics.
"""

from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional, Any


class SubredditQuery(BaseModel):
    """Parameters for querying subreddit data."""
    name: str # Name of the subreddit
    time_filter: str # Time period to analyze ('week', 'year', 'all')
    sort_by: str # Post sorting method (e.g., 'top')

    def to_composite_key(self) -> str:
        """Generate a unique key for the query."""
        return f"{self.name}#{self.time_filter}#{self.sort_by}"


class NamedEntity(BaseModel):
    """Named entity extracted from subreddit text with associated metrics."""
    name: str # Name of the entity 
    count: int # Number of times the entity appears in the text 
    sentiment: Optional[float] = None # Sentiment score for the entity 
    key_points: Optional[List[str]] = None # Key discussion points about the entity 
    urls: Optional[List[str]] = None # Related URLs


class NGram(BaseModel):
    """N-gram frequency data."""
    name: str # The n-gram text
    count: int # Number of times the n-gram appears in the text 


class SubredditAnalysis(BaseModel):
    timestamp: int # when the analysis was generated 
    num_words: int # Total word count analyzed 
    subreddit: str # Name of the subreddit 
    #  key = date
    top_n_grams: Dict[str, List[NGram]] # Top n-grams by date 
    # key = date 
    top_named_entities: Dict[str, List[NamedEntity]] # Top named entities by date 
    top_named_entities_embeddings: Dict[str, Tuple[float, float]] # Embeddings of the top named entities by date 
    top_named_entities_wordcloud: str # Wordcloud of the top named entities by date 
    readability_metrics: Dict[str, Any] # Readability metrics by date 

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
