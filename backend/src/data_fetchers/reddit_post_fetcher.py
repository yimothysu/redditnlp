"""Module for fetching and processing Reddit posts and their comments.

Provides data models and functions for retrieving post content, comments,
and associated metadata from Reddit using asyncpraw.
"""

import asyncpraw
from pydantic import BaseModel
from typing import Optional


class Comment(BaseModel):
    """Data model for a Reddit comment.
    
    Attributes:
        text: The comment's content
        score: The comment's score (upvotes - downvotes)
    """
    text: str
    score: int 


class RedditPost(BaseModel):
    """Data model for a Reddit post with its metadata and comments.
    
    Attributes:
        title: Post title
        description: Post content/selftext
        score: Post score (upvotes - downvotes)
        url: URL of the post
        created_utc: Post creation timestamp in UTC
        num_comments: Total number of comments
        comments: List of comment texts
        comment_scores: List of comment scores
    """
    title: str
    description: str 
    score: int
    url: str
    created_utc: float
    num_comments: int
    comments: list[str]
    comment_scores: list[int]
    
    def to_dict(self):
        """Convert the post data to a dictionary format."""
        return self.model_dump()


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
        post_comments = []  
        comment_scores = []
        
        # Extract comment text and scores, skipping MoreComments objects
        for comment in comments:
            if isinstance(comment, asyncpraw.models.MoreComments):
                continue
            if hasattr(comment, "body"):
                post_comments.append(comment.body)
            if hasattr(comment, "score"):
                comment_scores.append(comment.score)

        # Create Comment objects and sort by score
        comment_objects = [
            Comment(text=text, score=score) 
            for text, score in zip(post_comments, comment_scores)
        ]
        comment_objects.sort(key=lambda comment: comment.score)
        
        # sampling comments with the highest score if there's a large # of comments 
        if len(post_comments) > 30 and len(post_comments) <= 60:
            comment_objects = comment_objects[-25:] # most highly scored 25 comments 
        if len(post_comments) > 60 and len(post_comments) <= 100:
            comment_objects = comment_objects[-50:] # most highly scored 50 comments 
        if len(post_comments) > 100:
            comment_objects = comment_objects[-75:] # most highly scored 75 comments 

        post_comments = [comment_object.text for comment_object in comment_objects]
        comment_scores = [comment_object.score for comment_object in comment_objects]
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
    except Exception as e:
        print(f'Error fetching post data: {e}')
        return None
