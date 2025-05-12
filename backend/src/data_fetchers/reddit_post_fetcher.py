"""Module for fetching and processing Reddit posts and their comments.

Provides data models and functions for retrieving post content, comments,
and associated metadata from Reddit using asyncpraw.
"""
from __future__ import annotations
import asyncpraw, string 
from pydantic import BaseModel
from typing import Optional
from typing import List

class Comment(BaseModel):
    """Data model for a Reddit comment.
    
    Attributes:
        text: The comment's content
        score: The comment's score (upvotes - downvotes)
        replies: The comment's replies 
    """
    text: Optional[str] = None 
    # the context needed (title & description for top level comments) to resolve the pronouns in the comment 
    context: Optional[str] = None 
    score: Optional[int] = None 
    replies: Optional[List[Comment]] = None 


class RedditPost(BaseModel):
    """Data model for a Reddit post with its metadata and comments.
    
    Attributes:
        title: Post title
        description: Post content/selftext
        score: Post score (upvotes - downvotes)
        url: URL of the post
        created_utc: Post creation timestamp in UTC
        num_comments: Total number of comments
        top_level_comments: List of top level comment texts
        comment_scores: List of top level comment scores
    """
    title: str
    description: str 
    score: int
    permalink: str
    created_utc: float
    num_comments: int
    top_level_comments: list[Comment]
    
    def to_dict(self):
        """Convert the post data to a dictionary format."""
        return self.model_dump()

def ensure_ending_punctuation(text):
    ends_in_punctuation = text and text[-1] in string.punctuation
    if(not ends_in_punctuation):
        text = text.strip() + "."
    return text

def get_comment_replies(comment):
    # comment is of type praw.models.Comment
    # the replies field is of type praw.models.comment_forest.CommentForest
    replies = [] 
    for reply in comment.replies:
        reply_object = Comment() 
        if hasattr(reply, "body") and reply.body is not None and reply.body:
             reply_object.text = ensure_ending_punctuation(reply.body)
        else:
            continue 
        if hasattr(reply, "score") and reply.score is not None and reply.score > 0:
                reply_object.score = reply.score
        else:
            continue 
        if hasattr(reply, "replies") and reply.replies is not None and len(reply.replies) > 0:
             reply_object.replies = get_comment_replies(reply)
        else:
            reply_object.replies = []
        replies.append(reply_object)
    return replies 


async def fetch_post_data(post) -> Optional[RedditPost]:
    """Fetch and process data from a Reddit post including its comments.
    
    Retrieves post metadata and comments, filtering and sorting comments by score.
    For posts with many comments, only keeps the highest-scoring ones:
    - 25 comments for posts with 30-60 comments
    - 50 comments for posts with 60-100 comments
    - 75 comments for posts with >100 comments
    
    Args:
        post: A Reddit post object from asyncpraw
        
    Returns:
        RedditPost: A RedditPost object containing the post data and filtered comments
        None: If there was an error fetching the post data
    """
    try:
        # Fetch comments
        comments = await post.comments()
        comment_objects = []

        # Extract comment text and scores, skipping MoreComments objects
        for comment in comments:
            if isinstance(comment, asyncpraw.models.MoreComments):
                continue
            comment_object = Comment() 
            if hasattr(comment, "body") and comment.body is not None and comment.body:
                comment_object.text = ensure_ending_punctuation(comment.body) 
            else: 
                continue 
            if hasattr(comment, "score") and comment.score is not None and comment.score > 0:
                comment_object.score = comment.score 
            else:
                continue 
            if hasattr(comment, "replies") and comment.replies is not None and len(comment.replies) > 0:
                comment_object.replies = get_comment_replies(comment)
            else:
                comment_object.replies = []
            comment_objects.append(comment_object)

        print('got ', len(comment_objects), '/', len(comments), ' for post')
        return RedditPost(
            title=ensure_ending_punctuation(post.title),
            description=ensure_ending_punctuation(post.selftext),
            score=post.score,
            permalink=post.permalink,
            created_utc=post.created_utc,
            num_comments=post.num_comments,
            top_level_comments=comment_objects,
        )
    except Exception as e:
        print(f'Error fetching post data: {e}')
        return None
