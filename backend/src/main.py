"""FastAPI backend for Reddit analysis application.
Provides endpoints for subreddit analysis and popular subreddit retrieval."""

from http.client import BAD_REQUEST
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List
from email.message import EmailMessage

import asyncpraw
import os 
from dotenv import load_dotenv

from src.utils.subreddit_classes import (
    SubredditQuery,
    SubredditAnalysis,
)

from src.database.db import (
    fetch_subreddit_analysis,
)

from src.data_fetchers.toxicity_fetcher import get_toxicity_metrics 
from src.data_fetchers.positive_content_fetcher import get_positive_content_metrics

# Valid options for Reddit API queries
SORT_METHODS = ["top", "controversial"]
TIME_FILTERS = ["hour", "day", "week", "month", "year", "all"]

# App initialization
load_dotenv()
app = FastAPI(title="RedditNLP API")

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RedditPost(BaseModel):
    """Represents a Reddit post with its metadata and comments."""
    title: str
    description: str 
    score: int
    url: str
    created_utc: float
    num_comments: int
    comments: list[str]
    comment_scores: list[int]
    
    def to_dict(self):
        """Convert RedditPost to dictionary format."""
        return self.model_dump()


class Comment(BaseModel):
    """Represents a Reddit comment with its text and score."""
    text: str
    score: int


class SubredditRequest(BaseModel):
    """Request model for subreddit analysis requests."""
    subreddit: str
    email: str


@app.post("/analysis/", response_model=SubredditAnalysis)
async def fetch_subreddit_analysis_from_db(
    subreddit_query: SubredditQuery,
):
    """Fetch and return subreddit analysis with toxicity and positive content metrics.
    
    Raises:
        HTTPException: If time filter or sort method is invalid, or analysis not found.
    """
    # Input validation
    if subreddit_query.time_filter not in TIME_FILTERS:
        raise HTTPException(
            status_code=BAD_REQUEST,
            detail=f"Invalid time filter. Must be one of {TIME_FILTERS}.",
        )
    
    analysis = fetch_subreddit_analysis(subreddit_query)
    if not analysis:
        raise HTTPException(
            status_code=404, detail="Analysis not found for the given subreddit."
        )
    
    # Fetch and add toxicity and positive content metrics
    try:
        toxicity_score, toxicity_grade, toxicity_percentile, all_toxicity_scores, all_toxicity_grades = await get_toxicity_metrics(subreddit_query.name)
        positive_content_score, positive_content_grade, positive_content_percentile, all_positive_content_scores, all_positive_content_grades = await get_positive_content_metrics(subreddit_query.name)
        
        analysis.update({
            "toxicity_score": toxicity_score,
            "toxicity_grade": toxicity_grade,
            "toxicity_percentile": toxicity_percentile,
            "all_toxicity_scores": all_toxicity_scores,
            "all_toxicity_grades": all_toxicity_grades,
            "positive_content_score": positive_content_score,
            "positive_content_grade": positive_content_grade,
            "positive_content_percentile": positive_content_percentile,
            "all_positive_content_scores": all_positive_content_scores,
            "all_positive_content_grades": all_positive_content_grades
        })
    except:
        print("couldn't get toxicity and positive content metrics")
    
    return analysis


@app.post("/request_subreddit/", status_code=status.HTTP_204_NO_CONTENT)
async def request_subreddit(
    subreddit_request: SubredditRequest,
):
    """Handle new subreddit analysis requests and send notification email."""
    print(f"Subreddit Request from {subreddit_request.email} for the subreddit r/{subreddit_request.subreddit}")
    
    msg = EmailMessage()
    msg["Subject"] = "Subreddit Analysis Request"
    msg["From"] = "" # TODO
    msg["To"] = "" # TODO
    msg.set_content(
        f"A user with email {subreddit_request.email} has requested analysis for the subreddit r/{subreddit_request.subreddit}."
    )

    return Response(status_code=status.HTTP_204_NO_CONTENT)


class SubredditInfo(BaseModel):
    """Basic information about a subreddit."""
    name: str
    description: str


class PopularSubreddits(BaseModel):
    """Container for list of popular subreddits."""
    reddits: List[SubredditInfo]


@app.get("/popular_subreddits", response_model=PopularSubreddits)
async def get_popular_subreddits():
    """Fetch the top 5 most popular subreddits and their descriptions."""
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
