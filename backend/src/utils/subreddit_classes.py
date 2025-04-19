from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional, Any

class SubredditQuery(BaseModel):
    name: str
    time_filter: str
    sort_by: str

    def to_composite_key(self) -> str:
        return f"{self.name}#{self.time_filter}#{self.sort_by}"


class NamedEntity(BaseModel):
    name: str
    count: int 
    sentiment: Optional[float] = None 
    key_points: Optional[List[str]] = None 
    urls: Optional[List[str]] = None


class NGram(BaseModel):
    name: str
    count: int 


class SubredditAnalysis(BaseModel):
    timestamp: int # when the analysis was generated 
    num_words: int 
    subreddit: str
    #  key = date
    top_n_grams: Dict[str, List[NGram]]
    # key = date 
    top_named_entities: Dict[str, List[NamedEntity]]
    top_named_entities_embeddings: Dict[str, Tuple[float, float]]
    top_named_entities_wordcloud: str
    readability_metrics: Dict[str, Any]
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



