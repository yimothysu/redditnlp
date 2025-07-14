"""Module for calculating readability metrics of subreddit posts using various algorithms."""

from readability import Readability
from collections import defaultdict


def get_readability_metrics(posts):
    """Calculate readability metrics for a list of Reddit posts.
    
    Computes average word counts and readability scores using Flesch-Kincaid
    and Dale-Chall algorithms. Only posts with 100+ words are used for
    readability calculations.
    
    Args:
        posts: List of RedditPost objects
    Returns:
        dict: Dictionary containing:
            - avg_num_words_title: Average word count in titles
            - avg_num_words_description: Average word count in descriptions
            - avg_flesch_grade_level: Average Flesch-Kincaid grade level
            - dale_chall_grade_levels: Distribution of Dale-Chall grade levels
    """
    total_num_words_title = 0
    total_num_words_description = 0
    num_posts_analyzed = 0 
    readability_metrics = {
        "flesch_grade_level": 0.0,
        "dale_chall_grade_levels": defaultdict(int)
    }
    
    # Calculate word counts and readability scores
    for post in posts:
        num_words_title = len(post.title.split())
        num_words_description = len(post.description.split())
        total_num_words_title += num_words_title
        total_num_words_description += num_words_description
        
        try:
            r = Readability(post.title + post.description + "\n".join([top_level_comment.text for top_level_comment in post.top_level_comments[:10]]))
            flesch_kincaid = r.flesch_kincaid()
            readability_metrics["flesch_grade_level"] = float(flesch_kincaid.grade_level) + float(readability_metrics.get("flesch_grade_level", 0))
            dale_chall = r.dale_chall()
            for level in dale_chall.grade_levels:
                readability_metrics["dale_chall_grade_levels"][level] += 1
            num_posts_analyzed += 1 
        except Exception as e:
            print("An error occurred while generating readability metrics")
            print (f"Error: {e}")
    
    # Calculate averages
    avg_num_words_title = total_num_words_title / len(posts) if len(posts) != 0 else None
    avg_num_words_description = total_num_words_description / len(posts) if len(posts) != 0 else None
    avg_flesch_grade_level = readability_metrics.get("flesch_grade_level", 0) / num_posts_analyzed if num_posts_analyzed != 0 else -1
    
    return {
        "avg_num_words_title": avg_num_words_title,
        "avg_num_words_description": avg_num_words_description,
        "avg_flesch_grade_level": avg_flesch_grade_level,
        "dale_chall_grade_levels": readability_metrics["dale_chall_grade_levels"]
    }
