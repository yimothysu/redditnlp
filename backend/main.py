from http.client import BAD_REQUEST
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
import time 
import matplotlib.pyplot as plt

from reddit import get_comments, reddit
from plot_post_distribution import plot_post_distribution

app = FastAPI(title="RedditNLP API")
post_limit = 200

class RedditPost(BaseModel):
    title: str
    score: int
    url: str
    created_utc: float
    num_comments: int
    comments: list[str]

@app.get("/sample/{subreddit}", response_model=dict)
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
                created_utc=post.created_utc,
                num_comments=post.num_comments,
                comments= get_comments(post)
            )
            for post in posts
        ]
    sorted_slice_to_posts = slice_posts_list(posts_list, time_filter)
    #plot_post_distribution(subreddit, time_filter, sorted_slice_to_posts)
    return sorted_slice_to_posts

# posts_list is a list of RedditPost objects
# function returns a map where:
#   key = date of slice (Ex: 07/24)
#   value = vector of RedditPost objects for slice 
#
# if time_filter = year, slice by month 
# if time_filter = month, slice by week 
# if time_filter = week, slice by day 

def slice_posts_list(posts_list, time_filter):
    slice_to_posts = dict()
    for post in posts_list:
        # convert post date from utc to readable format 
        date_format = ''
        if time_filter == 'year': date_format = '%m-%y' # slicing a year by months 
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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
