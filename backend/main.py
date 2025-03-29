from http.client import BAD_REQUEST
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from datetime import datetime, timezone
import asyncpraw
import os 
import time 
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from praw.models import MoreComments
import asyncio
# TODO: Remove aiofiles if not needed
import aiofiles
import json
import random

#from reddit import get_comments, reddit
from plot_post_distribution import plot_post_distribution
from subreddit_nlp_analysis import get_subreddit_analysis, get_positive_content_metrics, get_toxicity_metrics

from word_embeddings import get_2d_embeddings
from word_cloud import generate_word_cloud

from analysis_cache import (
    SubredditQuery,
    SubredditAnalysis,
    SubredditAnalysisResponse,
    SubredditAnalysisCacheEntry,
    get_cache_entry,
    get_cache_lock
)


# Constants

SORT_METHODS = ["top", "controversial"]
TIME_FILTERS = ["hour", "day", "week", "month", "year", "all"]

TIME_FILTER_TO_POST_LIMIT = {'all': 400, 'year': 200, 'month': 100, 'week': 50}


# Initialization

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


# Classes

class RedditPost(BaseModel):
    title: str
    description: str 
    score: int
    url: str
    created_utc: float
    num_comments: int
    comments: list[str]
    comment_scores: list[int]
    
    # Turns RedditPost object into a dictionary
    def to_dict(self):
        return self.model_dump()

class Comment(BaseModel):
    text: str
    score: int

# class NamedEntity(BaseModel):
#     name: str 
#     count: int 
#     sentiment_score: float


# Methods

async def fetch_post_data(post):
    # get the post's comments 
    #print("inside fetch_post_data")
    try:
        comments = await post.comments()
        post_comments = []  
        comment_scores = []
        for comment in comments:
            if isinstance(comment, MoreComments): continue
            if hasattr(comment, "body"):
                post_comments.append(comment.body)
            if hasattr(comment, "score"):
                comment_scores.append(comment.score)

        # randomly sampling comments if there's a large # of comments 
        if len(post_comments) > 30 and len(post_comments) <= 60:
            post_comments = random.sample(post_comments, 25)
        if len(post_comments) > 60 and len(post_comments) <= 100:
            post_comments = random.sample(post_comments, 50)
        if len(post_comments) > 100:
            post_comments = random.sample(post_comments, 75)

        return RedditPost(
            title=post.title,
            description=post.selftext,
            score=post.score,
            url=post.url,
            created_utc=post.created_utc,
            num_comments=post.num_comments,
            comments=post_comments,
            comment_scores=comment_scores
        )
    except:
        print('could not get post comments')

async def print_to_json(posts_list, filename):
    # Save post to file
    # TODO: Save to database (maybe)
    try:
        json_data = json.dumps([post.to_dict() for post in posts_list], indent=4)
        async with aiofiles.open("posts.txt", "w") as f:
            await f.write(json_data)
        print("Posts saved to file")
    except Exception as e:
        print(f"Error saving posts to file: {e}")

async def perform_subreddit_analysis(cache_entry: SubredditAnalysisCacheEntry):
    def set_progress(progress: float):
        cache_entry.analysis_progress = progress

    # !!! WARNING !!!
    # sometimes PRAW will be down, or the # of posts you're trying to get at once 
    # is too much. PRAW will often return a 500 HTTP response in this case.
    # Try to: reduce post_limit, or wait a couple minutes

    subreddit_query = cache_entry.query
    
    reddit = asyncpraw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="reddit sampler",
    )

    post_limit = TIME_FILTER_TO_POST_LIMIT[subreddit_query.time_filter]

    cache_entry.analysis_status = "Fetching Post Data"

    subreddit_instance = await reddit.subreddit(subreddit_query.name)
    posts = [post async for post in subreddit_instance.top(
        limit=post_limit,
        time_filter=subreddit_query.time_filter
    )]

    # Fetch post data concurrently
    posts_list = await asyncio.gather(*(fetch_post_data(post) for post in posts))
    posts_list = [post for post in posts_list if post is not None]

    # Save post to file (uncomment to use)
    # await print_to_json(posts_list, "posts.txt")

    sorted_slice_to_posts = slice_posts_list(posts_list, subreddit_query.time_filter)

    print('Finished getting sorted_slice_to_posts')

    cache_entry.analysis_status = "Analyzing Post Data"

    #plot_post_distribution(subreddit, time_filter, sorted_slice_to_posts)
    top_n_grams, top_named_entities = await get_subreddit_analysis(sorted_slice_to_posts, set_progress)
    print(top_named_entities)

    t1 = time.time()
    entity_set= set()
    for _, entities in top_named_entities.items():
        entity_names = [entity[0] for entity in entities] # getting just the name of each entity 
        for entity_name in entity_names: entity_set.add(entity_name)
    print('# of elements in entity_set: ', len(entity_set))
    top_named_entities_embeddings = get_2d_embeddings(list(entity_set))
    print("# of entity embeddings: ", len(top_named_entities_embeddings))
    print(top_named_entities_embeddings)
    t2 = time.time()
    print('getting 2d embeddings took: ', t2 - t1)
    toxicity_score, toxicity_grade, toxicity_percentile, all_toxicity_scores, all_toxicity_grades = await get_toxicity_metrics(subreddit_query.name)
    positive_content_score, positive_content_grade, positive_content_percentile, all_positive_content_scores, all_positive_content_grades = await get_positive_content_metrics(subreddit_query.name)
    print('generating wordcloud: ')
    top_named_entities_wordcloud = generate_word_cloud(top_named_entities)
    print("wordcloud generated")
    
    analysis = SubredditAnalysis(
        top_n_grams = top_n_grams,
        top_named_entities = top_named_entities,
        top_named_entities_embeddings = top_named_entities_embeddings,
        top_named_entities_wordcloud = top_named_entities_wordcloud,

        # toxicity metrics 
        toxicity_score = toxicity_score,
        toxicity_grade = toxicity_grade,
        toxicity_percentile = toxicity_percentile,
        all_toxicity_scores = all_toxicity_scores,
        all_toxicity_grades = all_toxicity_grades,
        # positive content metrics 
        positive_content_score = positive_content_score,
        positive_content_grade = positive_content_grade,
        positive_content_percentile = positive_content_percentile,
        all_positive_content_scores = all_positive_content_scores,
        all_positive_content_grades = all_positive_content_grades       
    )
    #print(analysis)

    cache_entry.analysis = analysis
    cache_entry.updating = False


# API

@app.post("/analysis/", response_model=SubredditAnalysisResponse)
async def analyze_subreddit(
    subreddit_query: SubredditQuery,
    background_tasks: BackgroundTasks
):
    # Input validation
    if subreddit_query.time_filter not in TIME_FILTERS:
        raise HTTPException(
            status_code=BAD_REQUEST,
            detail=f"Invalid time filter. Must be one of {TIME_FILTERS}.",
        )
    if subreddit_query.sort_by not in SORT_METHODS:
        raise HTTPException(
            status_code=BAD_REQUEST,
            detail=f"Invalid sort method. Must be one of {SORT_METHODS}.",
        )

    # Check cache
    lock = await get_cache_lock(subreddit_query)
    async with lock:
        cache_entry = get_cache_entry(subreddit_query)
        if cache_entry.updating:
            return SubredditAnalysisResponse(
                analysis_status=cache_entry.analysis_status,
                analysis_progress=cache_entry.analysis_progress,
                analysis=None
            )
        if cache_entry.analysis is not None and not cache_entry.is_expired():
            return SubredditAnalysisResponse(
                analysis_status=None,
                analysis_progress=None,
                analysis=cache_entry.analysis
            )
        
        # Commence analysis
        cache_entry.updating = True
        cache_entry.last_update_timestamp = datetime.now(timezone.utc)
        cache_entry.analysis_status = "Commencing Analysis"
        cache_entry.analysis_progress = None

    background_tasks.add_task(perform_subreddit_analysis, cache_entry)

    return SubredditAnalysisResponse(
        analysis_status=cache_entry.analysis_status,
        analysis_progress=cache_entry.analysis_progress,
        analysis=None
    )


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
        else: date_format = '%m-%d' # slicing a month and week by days 
        post_date = datetime.fromtimestamp(post.created_utc).strftime(date_format)

        if post_date not in slice_to_posts:
            slice_to_posts[post_date] = []
        slice_to_posts[post_date].append(post)

    # sort slice_to_posts by date
    dates = list(slice_to_posts.keys())
    sorted_months = sorted(dates, key=lambda x: datetime.strptime(x, date_format))
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

    most_popular_reddits = reddit.subreddits.popular(limit=5)
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