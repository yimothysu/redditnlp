"""
Analyzes and grades subreddits based on their toxicity levels.
Provides metrics including toxicity scores, letter grades, and percentile rankings.
"""

from scipy.stats import percentileofscore


async def get_post_title_and_description(post):
    """Extract and combine post title and description text."""
    try:
        flattened_post_text = str(post.title) + "\n" + str(post.selftext)
        upvotes = post.score
        return (flattened_post_text, upvotes)
    except:
        print('could not get post title and description')


def get_percentile(values, target_value):
    """Calculate the percentile rank of a value within a distribution."""
    percentile = percentileofscore(values, target_value, kind='rank')
    return round(percentile, 4)


def get_toxicity_grade(toxicity):
    """Convert toxicity score to letter grade from A+ (least toxic) to F (most toxic)."""
    toxicity *= 100 # convert toxicity from [0, 1] scale to [0, 100] scale
    grade_to_toxicity_cutoff = {'A+': 0.05, 'A': 0.1, 'B+': 0.5, 'B': 1, 'B-': 5, 
                                'C+': 10, 'C': 15, 'C-': 20, 'D+': 30, 'D': 40, 'D-': 50}
    for grade, toxicity_cutoff in grade_to_toxicity_cutoff.items():
        if toxicity <= toxicity_cutoff: return grade
    return "F"


async def get_toxicity_metrics(sub_name):
    """
    Calculate toxicity metrics for a subreddit.
    Returns:
        - toxicity_score (float): Score from 0 (non-toxic) to 1 (toxic)
        - toxicity_grade (str): Letter grade
        - toxicity_percentile (float): Percentile ranking
        - all_toxicity_scores (list): Distribution of scores
        - all_toxicity_grades (list): Distribution of grades
    """
    toxicity_dict = {}
    with open("data/top_subreddits_toxicity_score.txt", "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" ") 
            if len(parts) == 2: 
                subreddit_name, score = parts
                toxicity_dict[subreddit_name] = round(float(score), 4)
    all_toxicity_scores = list(toxicity_dict.values())
    all_toxicity_grades = [get_toxicity_grade(toxicity_score) for toxicity_score in all_toxicity_scores]
    
    if (sub_name in toxicity_dict):
        print('toxicity score already in top_subreddits_toxicity_score.txt')
        toxicity_score = toxicity_dict[sub_name]
        return (toxicity_score, 
                get_toxicity_grade(toxicity_score), 
                get_percentile(all_toxicity_scores, toxicity_score), 
                all_toxicity_scores, 
                all_toxicity_grades)