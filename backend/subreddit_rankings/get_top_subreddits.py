import time 
import asyncpraw
import os 
from dotenv import load_dotenv

load_dotenv()

async def get_top_subreddits():
    reddit = asyncpraw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="reddit_api"
    )

    most_popular_reddits = reddit.subreddits.popular(limit=5000)
    t1 = time.time()
    f = open("top_subreddits.txt", "w", encoding="utf-8")
    reddits = []
    async for subreddit in most_popular_reddits: 
        reddits.append(subreddit.display_name)
    t2 = time.time()
    print('finished getting most_popular_reddits in ', t2-t1)
    await reddit.close() 

    for reddit in reddits:
        f.write(reddit + "\n")
    t3 = time.time()
    print('finished writing reddits to top_subreddits.txt in ', t3-t2)
    return reddits 