from http.client import BAD_REQUEST
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
import matplotlib.pyplot as plt

from reddit import get_comments, reddit
from plot_post_distribution import plot_post_distribution

app = FastAPI(title="RedditNLP API")
post_limit = 10000

class RedditPost(BaseModel):
    title: str
    score: int
    url: str
    created_utc: float
    # num_comments: int
    # comments: list[str]


# if time_filter = year, default view is to slice by month 
# if time_filter = month, default view is to slice by week 
# if time_filter = week, default view is to slice by day 
@app.get("/sample/{subreddit}", response_model=list)
async def sample_subreddit(
    subreddit: str, time_filter: str = "year", limit: int = post_limit, sort_by: str = "top"
):
    SORT_METHODS = ["top", "controversial"]
    TIME_FILTERS = ["hour", "day", "week", "month", "year", "all"]
    # if limit > 100:
    #     raise HTTPException(status_code=BAD_REQUEST, detail="Limit cannot exceed 100")

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

    subreddit_instance = reddit.subreddit(subreddit)

    if sort_by == "top":
        posts = subreddit_instance.top(limit=limit, time_filter=time_filter)
    elif sort_by == "controversial":
        posts = subreddit_instance.controversial(limit=limit, time_filter=time_filter)

    posts_list = [
        RedditPost(
            title=post.title,
            score=post.score,
            url=post.url,
            created_utc=post.created_utc
            #num_comments=post.num_comments,
            #comments=get_comments(post),
        )
        for post in posts
    ]

    plot_post_distribution(subreddit, time_filter, posts_list)
    return posts_list


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
