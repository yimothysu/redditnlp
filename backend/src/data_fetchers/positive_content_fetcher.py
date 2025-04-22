"""
Analyzes and grades subreddits based on their positive content scores.
Provides metrics including normalized scores, letter grades, and percentile rankings.
"""

from scipy.stats import percentileofscore
from dotenv import load_dotenv

load_dotenv()


def get_percentile(values, target_value):
    """Calculate the percentile rank of a value within a distribution."""
    percentile = percentileofscore(values, target_value, kind='rank')
    return round(percentile, 4)


def get_positive_content_grade(positive_content_score):
    """Convert positive content score to letter grade from A+ to F."""
    positive_content_score *= 100
    grade_to_positive_content_cutoff = {'A+': 92, 'A': 84, 'B+': 75, 'B': 68, 'B-': 60, 'C+': 52,
                                        'C': 43, 'C-': 35, 'D+': 26, 'D': 17, 'D-': 9}
    for grade, positive_content_cutoff in grade_to_positive_content_cutoff.items():
        if positive_content_score >= positive_content_cutoff: return grade
    return "F"


async def get_positive_content_metrics(sub_name):
    """
    Calculate positive content metrics for a subreddit.
    Returns:
        - positive_content_score (float): Normalized score [0-1]
        - positive_content_grade (str): Letter grade
        - positive_content_percentile (float): Percentile ranking
        - all_positive_content_scores (list): Distribution of scores
        - all_positive_content_grades (list): Distribution of grades
    """
    positive_content_dict = {}
    with open("data/top_subreddits_positive_content_score.txt", "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" ") 
            if len(parts) == 2: 
                subreddit_name, score = parts
                positive_content_dict[subreddit_name] = round(float(score), 4)
    all_positive_content_scores = list(positive_content_dict.values())
    all_positive_content_scores_normalized = [(x + 1) / 2 for x in all_positive_content_scores] # changes score range from [-1, 1] to [0, 1]
    all_positive_content_grades = [get_positive_content_grade(x) for x in all_positive_content_scores_normalized]
    positive_content_score = 0
    
    if (sub_name in positive_content_dict):
        print('positive content score already in top_subreddits_positive_content_score.txt')
        positive_content_score = positive_content_dict[sub_name]
        positive_content_score_normalized = (positive_content_score + 1) / 2 # changes score range from [-1, 1] to [0, 1]
        return (positive_content_score_normalized, 
                get_positive_content_grade(positive_content_score_normalized), 
                get_percentile(all_positive_content_scores_normalized, positive_content_score_normalized), 
                all_positive_content_scores_normalized,
                all_positive_content_grades)
