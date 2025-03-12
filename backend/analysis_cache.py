from typing import Optional, Dict, List, Tuple
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta


# Constants

ONE_HOUR = timedelta(hours=1)
ONE_DAY = timedelta(days=1)
ONE_WEEK = timedelta(days=7)
ONE_MONTH = timedelta(days=30)


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

"""
Cache for analyses with composite key of the following form.

Composite Key: "[subreddit_name]#[sort_method]#[time_filter]"
    Example: "ufl#top#week"
"""
subreddit_analysis_cache: Dict[str, SubredditAnalysisCacheEntry] = {}

def get_cache_entry(subreddit_query: SubredditQuery) -> SubredditAnalysisCacheEntry:
    composite_key = subreddit_query.to_composite_key()
    cache_entry = subreddit_analysis_cache.get(composite_key)
    if cache_entry:
        return cache_entry
    cache_entry = SubredditAnalysisCacheEntry(subreddit_query)
    subreddit_analysis_cache[composite_key] = cache_entry
    return cache_entry

