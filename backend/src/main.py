from http.client import BAD_REQUEST
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List

import asyncpraw
import os 
from dotenv import load_dotenv

from praw.models import MoreComments
# TODO: Remove aiofiles if not needed
import aiofiles
import json
import random

from src.utils.subreddit_classes import (
    SubredditQuery,
    SubredditAnalysis,
)

from src.database.db import (
    fetch_subreddit_analysis,
)

from src.data_fetchers.toxicity_fetcher import get_toxicity_metrics 
from src.data_fetchers.positive_content_fetcher import get_positive_content_metrics

# Constants

SORT_METHODS = ["top", "controversial"]
TIME_FILTERS = ["hour", "day", "week", "month", "year", "all"]


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

class SubredditRequest(BaseModel):
    subreddit: str
    email: str


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


# API

@app.post("/analysis/", response_model=SubredditAnalysis)
async def fetch_subreddit_analysis_from_db(
    subreddit_query: SubredditQuery,
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
    
    analysis = fetch_subreddit_analysis(subreddit_query)
    if not analysis:
        raise HTTPException(
            status_code=404, detail="Analysis not found for the given subreddit."
        )
    toxicity_score, toxicity_grade, toxicity_percentile, all_toxicity_scores, all_toxicity_grades = await get_toxicity_metrics(subreddit_query.name)
    positive_content_score, positive_content_grade, positive_content_percentile, all_positive_content_scores, all_positive_content_grades = await get_positive_content_metrics(subreddit_query.name)
    
    print(type(analysis))

    analysis["toxicity_score"] = toxicity_score 
    analysis["toxicity_grade"] = toxicity_grade 
    analysis["toxicity_percentile"] = toxicity_percentile 
    analysis["all_toxicity_scores"] = all_toxicity_scores 
    analysis["all_toxicity_grades"] = all_toxicity_grades 
    
    analysis["positive_content_score"] = positive_content_score 
    analysis["positive_content_grade"] = positive_content_grade 
    analysis["positive_content_percentile"] = positive_content_percentile 
    analysis["all_positive_content_scores"] = all_positive_content_scores
    analysis["all_positive_content_grades"] = all_positive_content_grades 
    
    return analysis

@app.post("/request_subreddit/", status_code=status.HTTP_204_NO_CONTENT)
async def request_subreddit(
    subreddit_request: SubredditRequest,
):
    # TODO: Implement
    print(f"Subreddit Request from {subreddit_request.email} for the subreddit r/{subreddit_request.subreddit}")
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)


"""
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
"""

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