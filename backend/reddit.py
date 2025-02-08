import os

from dotenv import load_dotenv
import praw

load_dotenv()

reddit = praw.Reddit(
    client_id=os.environ["REDDIT_CLIENT_ID"],
    client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    user_agent="reddit sampler",
)


def get_comments(post):
    comment_list = post.comments.list()
    return [comment.body for comment in comment_list]
