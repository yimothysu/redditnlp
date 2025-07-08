from src.analysis.groq import ( summarize_posts )
from src.utils.subreddit_classes import ( RedditPost )

# parameter posts is of type list[Post]
def get_topics(subreddit: str, posts: list[RedditPost]):
    # add indices onto posts 
    for i in range(len(posts)):
        post = posts[i]
        post.idx = i 
        posts[i] = post 

    topics = summarize_posts(subreddit, posts)
    # don't include topics that have < 3 posts associated 
    topics = {topic: posts for topic, posts in topics.items() if len(posts) >= 3}
    return topics 