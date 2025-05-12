"""Module for extracting and analyzing topics from Reddit posts and comments.

Provides functionality for:
- Fetching posts and comments from subreddits
- Organizing posts by time periods
- Extracting topics using NLP techniques
- Saving analysis results
"""

import asyncio
import os
import asyncpraw
import json
import datetime
from dotenv import load_dotenv
import time
from src.data_fetchers.reddit_post_fetcher import fetch_post_data

# Load environment variables for Reddit API access
load_dotenv()

# Maximum number of posts to fetch for each time filter
time_filter_to_post_limit = {'all': 400, 'year': 200, 'month': 100, 'week': 50}


def get_subreddit_topics(sorted_slice_to_posts, num_topics=10):
    """Extract topics from subreddit posts and comments for each time slice.
    
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


def slice_posts_list(posts_list, time_filter):
    """Organize posts into time slices based on their creation dates.
    
    Args:
        posts_list: List of Reddit posts
        time_filter: Time period ('all', 'year', 'week', etc.)
    Returns:
        dict: Mapping of date strings to lists of posts from that date
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


async def get_subreddit_data(subreddit_name, time_filter='week'):
    """Fetch and organize posts from a subreddit.
    
    Args:
        subreddit_name: Name of the subreddit to analyze
        time_filter: Time period to fetch posts from (default: 'week')
    Returns:
        dict: Posts organized by time slices
    Raises:
        ValueError: If time_filter is invalid
    """
    if time_filter not in ['hour', 'day', 'week', 'month', 'year', 'all']:
        raise ValueError(f"Invalid time filter. Must be one of ['hour', 'day', 'week', 'month', 'year', 'all'].")

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
        posts = [post async for post in subreddit_instance.top(limit=post_limit, time_filter=time_filter)]
    
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
    """Extract topics from a subreddit and output results.
    
    Fetches data, performs topic extraction, and saves results to a JSON file.
    
    Args:
        subreddit_name: Name of the subreddit to analyze
        time_filter: Time period to analyze (default: 'week')
        num_topics: Number of topics to extract (default: 10)
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
    """Command-line interface for topic extraction."""
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
    
    num_topics_input = input("Enter number of topics to extract (default is 10): ").strip()
    num_topics = int(num_topics_input) if num_topics_input.isdigit() else 10
    
    await extract_and_print_topics(subreddit_name, time_filter, num_topics)


if __name__ == "__main__":
    asyncio.run(main()) 