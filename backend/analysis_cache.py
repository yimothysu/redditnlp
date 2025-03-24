from typing import Optional, Dict, List, Tuple
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta
from asyncio import Lock


# Constants

ONE_HOUR = timedelta(hours=1)
ONE_DAY = timedelta(days=1)
ONE_WEEK = timedelta(days=7)
ONE_MONTH = timedelta(days=30)


# Classes

class SubredditQuery(BaseModel):
    name: str
    time_filter: str
    sort_by: str

    def to_composite_key(self) -> str:
        return f"{self.name}#{self.time_filter}#{self.sort_by}"

class SubredditAnalysis(BaseModel):
    #  key = date, value = list of top n grams for slice 
    # Tuple[str, int] = (n-gram, n-gram # occurrences)
    top_n_grams: Dict[str, List[Tuple[str, int]]]
    # key = date, value = list of top 10 named entities for slice 
    # Tuple[str, int, float, List[str]] = 
    #   1. named entity
    #   2. named entity # occurrences
    #   3. named entity sentiment score
    #   4. summary of comments regarding named entity 
    top_named_entities: Dict[str, List[Tuple[str, int, float, str]]]
    top_named_entities_embeddings: Dict[str, Tuple[float, float]]
    
    # For subreddit ranking
    toxicity_score: float # [0, 1] --> 0 = not toxic at all, 1 = all toxic 
    toxicity_grade: str # A+ to F 
    toxicity_percentile: float # [0, 100]
    all_toxicity_scores: List[float] # for generating a toxicity scores distribution graph on the UI 
    all_toxicity_grades: List[str] # for generating a toxicity grades distribution graph on the UI 

    positive_content_score: float # [0, 1] --> 0 = no positive content, 1 = all positive content
    positive_content_grade: str # A+ to F 
    positive_content_percentile: float # [0, 100]
    all_positive_content_scores: List[float] # for generating a positive content scores distribution graph on the UI 
    all_positive_content_grades: List[str] # for generating a positive content grades distribution graph on the UI 


class SubredditAnalysisResponse(BaseModel):
    analysis_status: Optional[str] = None
    analysis_progress: Optional[float] = None
    analysis: Optional[SubredditAnalysis] = None

class SubredditAnalysisCacheEntry():
    query: SubredditQuery
    analysis: SubredditAnalysis

    last_update_timestamp: datetime
    updating: bool
    analysis_status: str
    analysis_progress: float

    def __init__(self, query: SubredditQuery):
        self.query = query
        self.analysis = None
        self.last_update_timestamp = None
        self.updating = False
        self.analysis_status = None
        self.analysis_progress = None

    def is_expired(self) -> bool:
        delta = datetime.now(timezone.utc) - self.last_update_timestamp
        time_filter = self.query.time_filter
        if time_filter == "all" or time_filter == "year":
            return delta >= ONE_MONTH
        if time_filter == "month":
            return delta >= ONE_WEEK
        if time_filter == "week":
            return delta >= ONE_DAY
        return delta >= ONE_HOUR


# Cache setup

"""
Cache for analyses with composite key of the following form.

Composite Key: "[subreddit_name]#[sort_method]#[time_filter]"
    Example: "ufl#top#week"
"""
cache: Dict[str, SubredditAnalysisCacheEntry] = {}
cache_locks: Dict[str, Lock] = {}

async def get_cache_lock(query: SubredditQuery) -> Lock:
    key = query.to_composite_key()
    if key not in cache_locks:
        cache_locks[key] = Lock()
    return cache_locks[key]

def get_cache_entry(query: SubredditQuery) -> SubredditAnalysisCacheEntry:
    key = query.to_composite_key()
    cache_entry = cache.get(key)
    if cache_entry:
        return cache_entry
    cache_entry = SubredditAnalysisCacheEntry(query)
    cache[key] = cache_entry
    return cache_entry

