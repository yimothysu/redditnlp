import asyncpraw
from pydantic import BaseModel
from typing import Optional

class Comment(BaseModel):
    text: str
    score: int 

class RedditPost(BaseModel):
    title: str
    description: str 
    score: int
    url: str
    created_utc: float
    num_comments: int
    comments: list[str]
    comment_scores: list[int]
    
    def to_dict(self):
        return self.model_dump()

async def fetch_post_data(post) -> Optional[RedditPost]:
    """
    Fetch data from a Reddit post including its comments.
    
    Args:
        post: A Reddit post object from asyncpraw
        
    Returns:
        RedditPost: A RedditPost object containing the post data and filtered comments
        None: If there was an error fetching the post data
    """
    try:
        comments = await post.comments()
        post_comments = []  
        comment_scores = []
        for comment in comments:
            if isinstance(comment, asyncpraw.models.MoreComments):
                continue
            if hasattr(comment, "body"):
                post_comments.append(comment.body)
            if hasattr(comment, "score"):
                comment_scores.append(comment.score)

        comment_objects = []
        for i in range(len(post_comments)):
            comment_objects.append(Comment(text=post_comments[i], score=comment_scores[i]))
        # sort the comments by their score 
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