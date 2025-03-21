
import os
import asyncio
import time 
import asyncpraw
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

async def fetch_post_data(post):
    try:
        flattened_post_text = str(post.title) + "\n" + str(post.selftext)
        upvotes = post.score
        print('flattened_post_text: ', flattened_post_text)
        return (flattened_post_text, upvotes)
    except:
        print('could not get post title and description')

async def get_positive_content_score_for_each_sub(reddits):
    reddit = asyncpraw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="reddit_api"
    )
    analyzer = SentimentIntensityAnalyzer()

    f = open("top_subreddits_positive_content_score.txt", "a", encoding="utf-8")
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

            weighted_sentiment_sum = 0
            total_upvotes = 0
            for flattened_post_text, upvotes in posts:
                score = analyzer.polarity_scores(flattened_post_text)
                weighted_sentiment_sum += score['compound'] * upvotes
                total_upvotes += upvotes
            weighted_sentiment_score = weighted_sentiment_sum / total_upvotes
            print(r + " " + str(weighted_sentiment_score))
            f.write(r + " " + str(weighted_sentiment_score) + "\n")
        except Exception as e:
            print('could not get posts for r/' + r + ' because ', e)
    print('finished getting all ratios')
    await reddit.close() 

async def main():
    reddits = []
    skip = True
    with open("top_subreddits.txt", "r") as file:
        for line in file:
            if line.strip() == "investimentos":
                skip = False
            if not skip:
                reddits.append(line.strip()) 
    print('len(reddits): ', len(reddits))

    await get_positive_content_score_for_each_sub(reddits)

asyncio.run(main())