from pydantic import BaseModel
from typing import Dict, List, Tuple 

class SubredditQuery(BaseModel):
    name: str
    time_filter: str
    sort_by: str

    def to_composite_key(self) -> str:
        return f"{self.name}#{self.time_filter}#{self.sort_by}"

class SubredditAnalysis(BaseModel):
    subreddit: str
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
    top_named_entities_wordcloud: str

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

