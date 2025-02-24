from http.client import BAD_REQUEST
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Tuple

import datetime
import asyncpraw 
import os 
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from praw.models import MoreComments
import asyncio
# TODO: Remove aiofiles if not needed
import aiofiles
import json

#from reddit import get_comments, reddit
from plot_post_distribution import plot_post_distribution
from subreddit_nlp_analysis import get_subreddit_analysis

load_dotenv()
app = FastAPI(title="RedditNLP API")

origins = ['*']  # TODO: update

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

time_filter_to_post_limit = {'all': 400, 'year': 200, 'month': 100, 'week': 50}

class RedditPost(BaseModel):
    title: str
    score: int
    url: str
    created_utc: float
    num_comments: int
    comments: list[str]
    
    # Turns RedditPost object into a dictionary
    def to_dict(self):
        return self.model_dump()

class SubredditNLPAnalysis(BaseModel):
    #  key = date, value = list of top n grams for slice using sklearn's CountVectorizer
    top_n_grams: Dict[str, List[Tuple[str, int]]]
    # key = date, value = list of top named entities for slice using spaCy 
    top_named_entities: Dict[str, List[Tuple[str, int]]]

async def fetch_post_data(post):
    # get the post's comments 
    #print("inside fetch_post_data")
    try:
        comments = await post.comments()
        post_comments = []  
        for comment in comments:
            if isinstance(comment, MoreComments): continue
            if hasattr(comment, "body"):
                post_comments.append(comment.body)
    
    return RedditPost(
        title=post.title,
        score=post.score,
        url=post.url,
        created_utc=post.created_utc,
        num_comments=post.num_comments,
        comments=post_comments
    )

async def print_to_json(data, filename):
    # Save post to file
    # TODO: Save to database (maybe)
    try:
        json_data = json.dumps([post.to_dict() for post in posts_list], indent=4)
        async with aiofiles.open("posts.txt", "w") as f:
            await f.write(json_data)
        print("Posts saved to file")
    except Exception as e:
        print(f"Error saving posts to file: {e}")

@app.get("/sample/{subreddit}", response_model=SubredditNLPAnalysis)
async def sample_subreddit(
    subreddit: str, time_filter: str = "week", sort_by: str = "top"
):
    reddit = asyncpraw.Reddit(
    client_id=os.environ["REDDIT_CLIENT_ID"],
    client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    user_agent="reddit sampler",
    )   

    # !!! WARNING !!!
    # sometimes PRAW will be down, or the # of posts you're trying to get at once 
    # is too much. PRAW will often return a 500 HTTP response in this case.
    # Try to: reduce post_limit, or wait a couple minutes 

    post_limit = time_filter_to_post_limit[time_filter]
    SORT_METHODS = ["top", "controversial"]
    TIME_FILTERS = ["hour", "day", "week", "month", "year", "all"]

    if time_filter not in TIME_FILTERS:
        raise HTTPException(
            status_code=BAD_REQUEST,
            detail=f"Invalid time filter. Must be one of {TIME_FILTERS}.",
        )

    if sort_by not in SORT_METHODS:
        raise HTTPException(
            status_code=BAD_REQUEST,
            detail=f"Invalid sort method. Must be one of {SORT_METHODS}.",
        )

    subreddit_instance = await reddit.subreddit(subreddit)
    posts = [post async for post in subreddit_instance.top(limit=post_limit, time_filter=time_filter)]
    # Fetch post data concurrently
    posts_list = await asyncio.gather(*(fetch_post_data(post) for post in posts))
    
    # Save post to file (uncomment to use)
    # print_to_json(posts_list, "posts.txt")
    
    sorted_slice_to_posts = slice_posts_list(posts_list, time_filter)
    t4 = time.time()
    print('finished getting sorted_slice_to_posts in: ', t4-t3)
    #plot_post_distribution(subreddit, time_filter, sorted_slice_to_posts)
    top_n_grams, top_named_entities = get_subreddit_analysis(sorted_slice_to_posts)
    print(top_named_entities)
    analysis = SubredditNLPAnalysis(
        top_n_grams = top_n_grams,
        top_named_entities = top_named_entities
    )
    print(analysis)
    return analysis



# posts_list is a list of RedditPost objects
# function returns a sorted map by date where:
#   key = date of slice (Ex: 07/24)
#   value = vector of RedditPost objects for slice 

def slice_posts_list(posts_list, time_filter):
    slice_to_posts = dict()
    for post in posts_list:
        # convert post date from utc to readable format 
        date_format = ''
        if time_filter == 'all': date_format = '%y' # slicing all time by years 
        elif time_filter == 'year': date_format = '%m-%y' # slicing a year by months 
        else: date_format = '%d-%m' # slicing a month and week by days 
        post_date = datetime.datetime.fromtimestamp(post.created_utc).strftime(date_format)

        if post_date not in slice_to_posts:
            slice_to_posts[post_date] = []
        slice_to_posts[post_date].append(post)

    # sort slice_to_posts by date
    dates = list(slice_to_posts.keys())
    sorted_months = sorted(dates, key=lambda x: datetime.datetime.strptime(x, date_format))
    sorted_slice_to_posts = {i: slice_to_posts[i] for i in sorted_months}
    return sorted_slice_to_posts

class SubredditInfo(BaseModel):
    name: str
    description: str

class PopularSubreddits(BaseModel):
    reddits: List[SubredditInfo]

@app.get("/popular_subreddits", response_model=PopularSubreddits)
async def get_popular_subreddits():
    
    reddit = asyncpraw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="reddit_api"
    )

    most_popular_reddits = reddit.subreddits.popular(limit=10)
    reddits = []
    async for subreddit in most_popular_reddits:
        reddits.append(
            SubredditInfo(
                name=subreddit.display_name,
                description=subreddit.public_description,
                )
            )
    await reddit.close()
    return PopularSubreddits(reddits=reddits)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)