import asyncio
import os
import asyncpraw
import json
import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Tuple
import time

# Need to load environment variables for Reddit API access
load_dotenv()

# Define the RedditPost model similar to the one in main.py
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

class Comment(BaseModel):
    text: str
    score: int 


# Time filter to post limit mapping
time_filter_to_post_limit = {'all': 400, 'year': 200, 'month': 100, 'week': 50}

def get_subreddit_topics(sorted_slice_to_posts, num_topics=10):
    """
    Extract topics from subreddit posts and comments for each time slice.
    
    Args:
        sorted_slice_to_posts (dict): Dictionary mapping time slices to posts
        num_topics (int, optional): Number of topics to extract per slice. Defaults to 10.
        
    Returns:
        dict: Dictionary mapping time slices to topic extraction results
    """
    t1 = time.time()
    subreddit_topics = {}
    
    for date, posts in sorted_slice_to_posts.items():
        # Combine post titles and comments for topic modeling
        texts = []
        for post in posts:
            # Add post title (titles are often more informative than comments)
            if post.title and len(post.title.strip()) > 10:  # Only add substantial titles
                texts.append(post.title)
            
            # Process and add comments
            filtered_comments = []
            for comment in post.comments:
                # Skip very short comments or empty comments
                if comment and len(comment.strip()) > 15:
                    filtered_comments.append(comment)
            
            # Add filtered comments
            texts.extend(filtered_comments)
        
        # Skip if there are not enough texts for meaningful topic modeling
        if len(texts) < 5:
            continue
            
        # Extract topics from the texts
        topics_result = extract_topics(texts, num_topics)
        subreddit_topics[date] = topics_result
    
    t2 = time.time()
    print('finished topic extraction in: ', t2-t1)
    
    return subreddit_topics


async def fetch_post_data(post):
    """Fetch data from a Reddit post including its comments"""
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

def slice_posts_list(posts_list, time_filter):
    """
    Organize posts by time slices based on the time filter
    Returns a dictionary where keys are date strings and values are lists of posts
    """
    slice_to_posts = dict()
    for post in posts_list:
        # Convert post date from UTC to readable format
        date_format = ''
        if time_filter == 'all': 
            date_format = '%y'  # Slicing all time by years
        elif time_filter == 'year': 
            date_format = '%m-%y'  # Slicing a year by months
        else: 
            date_format = '%m-%d'  # Slicing a month and week by days
        
        post_date = datetime.datetime.fromtimestamp(post.created_utc).strftime(date_format)

        if post_date not in slice_to_posts:
            slice_to_posts[post_date] = []
        slice_to_posts[post_date].append(post)

    # Sort slice_to_posts by date
    dates = list(slice_to_posts.keys())
    sorted_months = sorted(dates, key=lambda x: datetime.datetime.strptime(x, date_format))
    sorted_slice_to_posts = {i: slice_to_posts[i] for i in sorted_months}
    return sorted_slice_to_posts

async def get_subreddit_data(subreddit_name, time_filter='week', sort_by='top'):
    """Fetch data from a subreddit"""
    if time_filter not in ['hour', 'day', 'week', 'month', 'year', 'all']:
        raise ValueError(f"Invalid time filter. Must be one of ['hour', 'day', 'week', 'month', 'year', 'all'].")
    
    if sort_by not in ['top', 'controversial']:
        raise ValueError(f"Invalid sort method. Must be one of ['top', 'controversial'].")
    
    # Initialize Reddit API client
    reddit = asyncpraw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="Reddit Topic Extractor",
    )
    
    try:
        post_limit = time_filter_to_post_limit[time_filter]
        
        # Get subreddit instance
        subreddit_instance = await reddit.subreddit(subreddit_name)
        
        # Get posts based on sort method
        if sort_by == 'top':
            posts = [post async for post in subreddit_instance.top(limit=post_limit, time_filter=time_filter)]
        else:  # controversial
            posts = [post async for post in subreddit_instance.controversial(limit=post_limit, time_filter=time_filter)]
        
        # Fetch post data concurrently
        posts_list = await asyncio.gather(*(fetch_post_data(post) for post in posts))
        posts_list = [post for post in posts_list if post is not None]  # Filter out None values
        
        # Organize posts by time slices
        sorted_slice_to_posts = slice_posts_list(posts_list, time_filter)
        
        return sorted_slice_to_posts
    finally:
        # Close the Reddit client
        await reddit.close()

async def extract_and_print_topics(subreddit_name, time_filter='week', num_topics=10):
    """
    Extract topics from a subreddit and print the results
    """
    print(f"Fetching data from r/{subreddit_name} (time filter: {time_filter})...")
    
    # Get subreddit data
    sorted_slice_to_posts = await get_subreddit_data(subreddit_name, time_filter)
    
    print(f"Found posts in {len(sorted_slice_to_posts)} time slices")
    
    # Extract topics from the subreddit data
    print(f"Extracting topics...")
    subreddit_topics = get_subreddit_topics(sorted_slice_to_posts, num_topics)
    
    # Print the extracted topics
    print(f"\nExtracted Topics for r/{subreddit_name} (time filter: {time_filter}):")
    for date, topics_result in subreddit_topics.items():
        print(f"\n--- Time Slice: {date} ---")
        
        if "error" in topics_result:
            print(f"Error: {topics_result['error']}")
            continue
        
        print("Top Topics:")
        for topic_id, topic_words in topics_result["topic_representations"].items():
            print(f"Topic {topic_id}:")
            for word, weight in topic_words[:5]:  # Show top 5 words per topic
                print(f"  - {word}: {weight:.4f}")
    
    # Save the results to a JSON file
    output_file = f"{subreddit_name}_{time_filter}_topics.json"
    with open(output_file, 'w') as f:
        # Convert to a serializable format
        serializable_data = {}
        for date, topics_result in subreddit_topics.items():
            if "topic_info" in topics_result:
                # Convert DataFrame records to a serializable format
                topics_result["topic_info"] = [dict(item) for item in topics_result["topic_info"]]
            serializable_data[date] = topics_result
        
        json.dump(serializable_data, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

async def main():
    # Get user input for subreddit and time filter
    subreddit_name = input("Enter a subreddit name (without r/): ").strip()
    
    print("\nTime filter options:")
    print("1. hour")
    print("2. day")
    print("3. week (default)")
    print("4. month")
    print("5. year")
    print("6. all")
    
    time_filter_choice = input("Choose time filter (1-6, default is 3): ").strip()
    
    # Map choice to actual time filter
    time_filter_map = {
        '1': 'hour',
        '2': 'day',
        '3': 'week',
        '4': 'month',
        '5': 'year',
        '6': 'all'
    }
    
    time_filter = time_filter_map.get(time_filter_choice, 'week')
    
    # Get number of topics
    num_topics_input = input("Enter number of topics to extract (default is 10): ").strip()
    num_topics = int(num_topics_input) if num_topics_input.isdigit() else 10
    
    # Extract and print topics
    await extract_and_print_topics(subreddit_name, time_filter, num_topics)

if __name__ == "__main__":
    asyncio.run(main()) 