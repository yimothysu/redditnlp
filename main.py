from http.client import BAD_REQUEST
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from reddit import reddit


app = FastAPI(title="RedditNLP API")


class RedditPost(BaseModel):
    title: str
    score: int
    url: str
    created_utc: float
    num_comments: int


@app.get("/sample/{subreddit}", response_model=list)
async def sample_subreddit(
    subreddit: str, time_filter: str = "week", limit: int = 10, sort_by: str = "top"
):
    SORT_METHODS = ["top", "controversial"]
    TIME_FILTERS = ["hour", "day", "week", "month", "year", "all"]
    if limit > 100:
        raise HTTPException(status_code=BAD_REQUEST, detail="Limit cannot exceed 100")

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
            created_utc=post.created_utc,
            num_comments=post.num_comments,
        )
        for post in posts
    ]

    return posts_list


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
