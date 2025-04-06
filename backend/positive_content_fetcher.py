from scipy.stats import percentileofscore
import os 
import asyncio
import asyncpraw
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer

load_dotenv()

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

def get_positive_content_grade(positive_content_score):
    # A+ means very positive content subreddit
    # F means very negative content subreddit 
    positive_content_score *= 100 # convert positive_content from [0, 1] scale to [0, 100] scale
    grade_to_positive_content_cutoff = {'A+': 92, 'A': 84, 'B+': 75, 'B': 68, 'B-': 60, 'C+': 52,
                                        'C': 43, 'C-': 35, 'D+': 26, 'D': 17, 'D-': 9}
    for grade, positive_content_cutoff in grade_to_positive_content_cutoff.items():
        if positive_content_score >= positive_content_cutoff: return grade
    return "F"

# This function will compute and return 4 values:
#   1. positive_content_score: float # [0, 1] --> 0 = no positive content, 1 = all positive content
#   2. positive_content_grade: str # A+ to F 
#   3. positive_content_percentile: float # [0, 100]
#   4. all_positive_content_scores: List[float] # for generating a positive content scores distribution graph on the UI 
#   5. all_positive_content_grades: List[str] # for generating a positive content grades distribution graph on the UI 
async def get_positive_content_metrics(sub_name):
    positive_content_dict = {}
    with open("top_subreddits_positive_content_score.txt", "r", encoding="utf-8") as file:
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
    else:
        reddit = asyncpraw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="reddit_api"
        )
        analyzer = SentimentIntensityAnalyzer()

        try:
            subreddit_instance = await reddit.subreddit(sub_name)
            posts = [post async for post in subreddit_instance.top(
                limit=400,
                time_filter="all"
            )]
            posts = await asyncio.gather(*(get_post_title_and_description(post) for post in posts))
            posts = [post for post in posts if post is not None]

            weighted_sentiment_sum = 0
            total_upvotes = 0
            for flattened_post_text, upvotes in posts:
                score = analyzer.polarity_scores(flattened_post_text)
                weighted_sentiment_sum += score['compound'] * upvotes
                total_upvotes += upvotes
            positive_content_score = weighted_sentiment_sum / total_upvotes
            positive_content_score_normalized = (positive_content_score + 1) / 2
            await reddit.close() 

            f = open("top_subreddits_positive_content_score.txt", "a", encoding="utf-8")
            f.write(sub_name + " " + str(positive_content_score) + "\n")
            print('wrote positive content score for r/', sub_name, ' to top_subreddits_positive_content_score.txt')
            all_positive_content_scores_normalized.append(positive_content_score_normalized)
            all_positive_content_grades.append(get_positive_content_grade(positive_content_score_normalized))
            return (positive_content_score_normalized, 
                    get_positive_content_grade(positive_content_score_normalized), 
                    get_percentile(all_positive_content_scores_normalized, positive_content_score_normalized), 
                    all_positive_content_scores_normalized,
                    all_positive_content_grades)
        except Exception as e:
            print('couldnt get posts to calculate positive content score because ', e)
