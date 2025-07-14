from src.analysis.groq import ( summarize_posts )
from src.utils.subreddit_classes import ( RedditPost )

def is_list_of_reddit_posts(obj):
    return (
        isinstance(obj, list)
        and all(isinstance(item, RedditPost) for item in obj)
    )

# parameter posts is of type list[Post]
def get_topics(subreddit: str, posts: list[RedditPost]):
    if not is_list_of_reddit_posts(posts):
        raise ValueError("Invalid input into get_topics: expected posts to be a list of RedditPost objects")
    # add indices onto posts 
    for i in range(len(posts)):
        post = posts[i]
        post.idx = i 
        posts[i] = post 
    
    topics = summarize_posts(subreddit, posts)
    # don't include topics that have < 3 posts associated 
    topics = {topic: posts for topic, posts in topics.items() if len(posts) >= 3}
    return topics 