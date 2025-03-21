
import os
import asyncio
import time 
import asyncpraw
from dotenv import load_dotenv
from detoxify import Detoxify

load_dotenv()

weights = {
        'toxicity': 1.0,
        'severe_toxicity': 1.5,  
        'obscene': 0.8,
        'threat': 1.5,  
        'insult': 1.0,
        'identity_attack': 1.2
}


async def fetch_post_data(post):
    try:
        flattened_post_text = str(post.title) + "\n" + str(post.selftext)
        upvotes = post.score
        #print('flattened_post_text: ', flattened_post_text)
        return (flattened_post_text, upvotes)
    except:
        print('could not get post title and description')

def composite_toxicity(text):
    t1 = time.time()
    scores = Detoxify('original').predict(text)
    print(scores)
    total_weight = sum(weights.values())
    composite_toxicity_score = round(sum(scores[k] * weights[k] for k in scores) / total_weight, 6)
    t2 = time.time()
    print('composite toxicity score of ', composite_toxicity_score, ' took ', t2-t1, ' to compute')
    return composite_toxicity_score


async def get_weighted_toxicity_score_for_each_sub(reddits):
    reddit = asyncpraw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="reddit_api"
    )
    
    f = open("top_subreddits_toxicity_score.txt", "a", encoding="utf-8")
    for r in reddits:
        try:
            subreddit_instance = await reddit.subreddit(r)
            posts = [post async for post in subreddit_instance.top(
                limit=400,
                time_filter="all"
            )]
            posts = await asyncio.gather(*(fetch_post_data(post) for post in posts))
            posts = [post for post in posts if post is not None]
            print('got ', len(posts), ' posts for r/', r)

            # all_weighted_toxicity_scores = await asyncio.gather(*(weighted_composite_toxicity(flattened_post_text, upvotes) for flattened_post_text, upvotes in posts))
            # weighted_toxicity_sum = [weighted_toxicity_score for weighted_toxicity_score in all_weighted_toxicity_scores]
            # total_upvotes = sum([upvotes for _, upvotes in posts])
            
            # weighted_toxicity_score = weighted_toxicity_sum / total_upvotes
            # print(r + " " + str(weighted_toxicity_score))
            # f.write(r + " " + str(weighted_toxicity_score) + "\n")

            all_flattened_post_text = "\n".join([flattened_post_text for flattened_post_text, _ in posts])
            composite_toxicity_score = composite_toxicity(all_flattened_post_text)
            print(r + " " + str(composite_toxicity_score))
            f.write(r + " " + str(composite_toxicity_score) + "\n")
        
        except Exception as e:
            print('could not get posts for r/' + r + ' because ', e)
    print('finished getting all toxicity scores')
    await reddit.close() 


async def main():
    reddits = []
    with open("top_subreddits.txt", "r") as file:
        for line in file:
            reddits.append(line.strip())  
    print('len(reddits): ', len(reddits))
    await get_weighted_toxicity_score_for_each_sub(reddits)

asyncio.run(main())