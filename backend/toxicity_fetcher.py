from scipy.stats import percentileofscore
import asyncio
import os 
import asyncpraw
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from detoxify import Detoxify

load_dotenv()

async def get_post_title_and_description(post):
    try:
        flattened_post_text = str(post.title) + "\n" + str(post.selftext)
        upvotes = post.score
        return (flattened_post_text, upvotes)
    except:
        print('could not get post title and description')

def composite_toxicity(text):
    weights = {
        'toxicity': 1.0,
        'severe_toxicity': 1.5,  
        'obscene': 0.8,
        'threat': 1.5,  
        'insult': 1.0,
        'identity_attack': 1.2
    }
    scores = Detoxify('original').predict(text)
    total_weight = sum(weights.values())
    composite_toxicity_score = round(sum(scores[k] * weights[k] for k in scores) / total_weight, 6)
    return composite_toxicity_score

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
    with open("top_subreddits_toxicity_score.txt", "r", encoding="utf-8") as file:
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
    else:
        reddit = asyncpraw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="reddit_api"
        )
        try:
            subreddit_instance = await reddit.subreddit(sub_name)
            posts = [post async for post in subreddit_instance.top(
                limit=400,
                time_filter="all"
            )]
            posts = await asyncio.gather(*(get_post_title_and_description(post) for post in posts))
            posts = [post for post in posts if post is not None]

            all_flattened_post_text = "\n".join([flattened_post_text for flattened_post_text, _ in posts])
            composite_toxicity_score = composite_toxicity(all_flattened_post_text)
            await reddit.close() 
            
            f = open("top_subreddits_toxicity_score.txt", "a", encoding="utf-8")
            f.write(sub_name + " " + str(composite_toxicity_score) + "\n")
            print('wrote toxicity score for r/', sub_name, ' to top_subreddits_toxicity_score.txt')
            all_toxicity_scores.append(composite_toxicity_score)
            all_toxicity_grades.append(get_toxicity_grade(composite_toxicity_score))
            return (composite_toxicity_score, 
                    get_toxicity_grade(composite_toxicity_score), 
                    get_percentile(all_toxicity_scores, composite_toxicity_score), 
                    all_toxicity_scores,
                    all_toxicity_grades)
        except Exception as e:
            print('couldnt get posts to calculate toxicity score because ', e)
