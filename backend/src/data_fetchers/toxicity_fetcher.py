from scipy.stats import percentileofscore


async def get_post_title_and_description(post):
    try:
        flattened_post_text = str(post.title) + "\n" + str(post.selftext)
        upvotes = post.score
        return (flattened_post_text, upvotes)
    except:
        print('could not get post title and description')


def get_percentile(values, target_value):
    # gets the percentile of target_value in values distribution 
    percentile = percentileofscore(values, target_value, kind='rank')
    return round(percentile, 4)


def get_toxicity_grade(toxicity):
    # A+ means extremely civil, respectful, and non-toxic subreddit.
    # F- means extremely toxic subreddit 
    toxicity *= 100 # convert toxicity from [0, 1] scale to [0, 100] scale
    grade_to_toxicity_cutoff = {'A+': 0.05, 'A': 0.1, 'B+': 0.5, 'B': 1, 'B-': 5, 
                                'C+': 10, 'C': 15, 'C-': 20, 'D+': 30, 'D': 40, 'D-': 50}
    for grade, toxicity_cutoff in grade_to_toxicity_cutoff.items():
        if toxicity <= toxicity_cutoff: return grade
    return "F"


# This function will compute and return 4 values:
#   1. toxicity_score: float # [0, 1] --> 0 = not toxic at all, 1 = all toxic 
#   2. toxicity_grade: str # A+ to F 
#   3. toxicity_percentile: float # [0, 100]
#   4. all_toxicity_scores: List[float] # for generating a toxicity scores distribution graph on the UI 
#   5. all_toxicity_grades: List[str] # for generating a toxicity grades distribution graph on the UI 
async def get_toxicity_metrics(sub_name):
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