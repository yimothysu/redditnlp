import os

from dotenv import load_dotenv
import praw
from praw.models import MoreComments

load_dotenv()

praw.__dict__["_warn_async"] = False

reddit = praw.Reddit(
    client_id=os.environ["REDDIT_CLIENT_ID"],
    client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    user_agent="reddit sampler",
)

# too slow 
# for getting the comments of ~1000 posts, i was waiting for minutes, and then eventually 
# it just crashed with a TooManyRequests error 
# I just realized using try and except is TOO slow 

def get_comments(post):
    # Getting just the first 20 comments to speed up program 
    post_comments = []
    post.comments.replace_more(limit=0)
    for comment in post.comments[slice(20)]:
        if isinstance(comment, MoreComments): continue
        post_comments.append(comment.body)
    print('len(post_comments): ', len(post_comments))
    return post_comments 
