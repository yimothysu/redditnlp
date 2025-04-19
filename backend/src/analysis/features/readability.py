from readability import Readability
from collections import defaultdict

def get_readability_metrics(posts):
    total_num_words_title = 0
    total_num_words_description = 0
    num_posts_over_100_words = 0
    readability_metrics = {
        "flesch_grade_level": 0.0,
        "dale_chall_grade_levels": defaultdict(int)
    }
    for post in posts:
        num_words_title = len(post.title.split())
        num_words_description = len(post.description.split())
        total_num_words_title += num_words_title
        total_num_words_description += num_words_description
        if num_words_description >= 100:
            num_posts_over_100_words += 1
            try:
                r = Readability(post.description)
                flesch_kincaid = r.flesch_kincaid()
                readability_metrics["flesch_grade_level"] = float(flesch_kincaid.grade_level) + float(readability_metrics.get("flesch_grade_level", 0))
                dale_chall = r.dale_chall()
                for level in dale_chall.grade_levels:
                    readability_metrics["dale_chall_grade_levels"][level] += 1
            except:
                print("An error occurred while generating readability metrics")
    avg_num_words_title = total_num_words_title / len(posts) if len(posts) != 0 else None
    avg_num_words_description = total_num_words_description / len(posts) if len(posts) != 0 else None
    avg_flesch_grade_level =  readability_metrics.get("flesch_grade_level", 0) / num_posts_over_100_words if num_posts_over_100_words != 0 else -1
    return {
             "avg_num_words_title": avg_num_words_title,
             "avg_num_words_description": avg_num_words_description,
             "avg_flesch_grade_level": avg_flesch_grade_level,
             "dale_chall_grade_levels": readability_metrics["dale_chall_grade_levels"]
            }