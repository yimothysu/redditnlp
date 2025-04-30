"""Module for performing comprehensive NLP analysis on subreddit content.
Includes functionality for post collection, text analysis, and feature extraction."""

import time, asyncio, asyncpraw, os, random  # type: ignore
from dotenv import load_dotenv # type: ignore
load_dotenv()
from datetime import datetime

from src.data_fetchers.reddit_topic_extractor import ( fetch_post_data )
from src.utils.subreddit_classes import ( SubredditQuery, SubredditAnalysis )
from src.analysis.features.named_entities import ( get_top_entities )
from src.analysis.features.n_grams import ( get_top_ngrams )
from src.analysis.features.readability import ( get_readability_metrics )
from src.analysis.features.word_embeddings import ( get_2d_embeddings )
from src.analysis.features.word_cloud import ( generate_word_cloud )

# Configuration maps for different time filters
TIME_FILTER_TO_INITIAL_POST_QUERY_LIMIT = {'all': 1000, 'year': 1000, 'week': 1000}
TIME_FILTER_TO_POST_LIMIT = {'all': 350, 'year': 200, 'week': 200}
TIME_FILTER_TO_NAMED_ENTITIES_LIMIT = {'all': 12, 'year': 10, 'week': 8}
TIME_FILTER_TO_DATE_FORMAT = {'all': '%y', 'year': '%m-%y', 'week': '%m-%d'}

config = {
    "time_filter": "week",
    "max_post_len": 1000000 # in characters 
}


async def get_post_title_and_description(post):
    """Extract title and description text from a Reddit post.
    
    Args:
        post: Reddit post object
    Returns:
        tuple: (combined post text, upvotes)
    """
    try:
        flattened_post_text = str(post.title) + "\n" + str(post.selftext)
        upvotes = post.score
        return (flattened_post_text, upvotes)
    except Exception as e:
        print('could not get post title and description: ', e)


def group_posts_by_date(posts):
    """Group Reddit posts by their creation date.
    
    Args:
        posts: List of RedditPost objects
    Returns:
        dict: Sorted map where key=date string, value=list of posts from that date
    """
    posts_grouped_by_date = dict()
    date_format = TIME_FILTER_TO_DATE_FORMAT[config['time_filter']]
    for post in posts:
        post_date = datetime.fromtimestamp(post.created_utc).strftime(date_format)

        if post_date not in posts_grouped_by_date:
            posts_grouped_by_date[post_date] = []
        posts_grouped_by_date[post_date].append(post)

    # Sort posts by date
    dates = list(posts_grouped_by_date.keys())
    sorted_months = sorted(dates, key=lambda x: datetime.strptime(x, date_format))
    posts_grouped_by_date = {i: posts_grouped_by_date[i] for i in sorted_months}
    return posts_grouped_by_date


async def get_subreddit_analysis(posts_grouped_by_date, comment_and_score_pairs_grouped_by_date):
    """Perform NLP analysis on grouped posts.
    
    Args:
        posts_grouped_by_date: Dictionary of posts grouped by date
    Returns:
        tuple: (top n-grams, top named entities)
    """
    post_content_grouped_by_date = dict() 
    for date, posts in posts_grouped_by_date.items():
        post_content = ' '.join([post.title + ' ' + post.description + ' '.join(post.comments) for post in posts])
        if len(post_content) >= config['max_post_len']:
            # randomly sample the comments if the post content is too long 
            comments = []
            for post in posts:
                half_of_comments = random.sample(post.comments, k=len(comments) // 2)
                comments.extend(half_of_comments)
            post_content = ' '.join([post.title + ' ' + post.description + ' '.join(comments)])
        post_content_grouped_by_date[date] = post_content
    
    top_n_grams = get_top_ngrams(post_content_grouped_by_date)
    top_named_entities = await get_top_entities(config['time_filter'], post_content_grouped_by_date, posts_grouped_by_date, comment_and_score_pairs_grouped_by_date)
    
    # Generate embeddings and word cloud for named entities
    entity_set = set()
    for _, entities in top_named_entities.items():
        entity_names = [entity.name for entity in entities]
        for entity_name in entity_names: entity_set.add(entity_name)
    top_named_entities_embeddings = get_2d_embeddings(list(entity_set))
    top_named_entities_wordcloud = generate_word_cloud(top_named_entities)
    
    return top_n_grams, top_named_entities, top_named_entities_embeddings, top_named_entities_wordcloud


async def perform_subreddit_analysis(subreddit_query: SubredditQuery):
    """Main function to perform comprehensive analysis on a subreddit.
    
    Fetches posts, analyzes content, and generates various NLP metrics.
    
    Args:
        subreddit_query: SubredditQuery object containing analysis parameters
    Returns:
        SubredditAnalysis: Complete analysis results
    """
    config['time_filter'] = subreddit_query.time_filter

    reddit = asyncpraw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="reddit sampler",
    )

    # Fetch initial set of posts
    initial_post_query_limit = TIME_FILTER_TO_INITIAL_POST_QUERY_LIMIT[subreddit_query.time_filter]
    subreddit_instance = await reddit.subreddit(subreddit_query.name)
    posts = [post async for post in subreddit_instance.top(
        limit=initial_post_query_limit,
        time_filter=subreddit_query.time_filter
    )]

    # Limit posts and prioritize those with more content
    if(len(posts) > TIME_FILTER_TO_POST_LIMIT[subreddit_query.time_filter]):
        posts.sort(key=lambda post: len(post.selftext) + len(post.title))
        posts = posts[-TIME_FILTER_TO_POST_LIMIT[subreddit_query.time_filter]:]

    '''
    if the number of posts is >= 200 and <= 300, then divide the posts list in half and wait a minute in between 
    querying the comments for the 1st batch and the 2nd batch (to avoid going over asyncpraw's api limit)
    
    if the number of posts is >= 300, then divide the posts list into thirds and wait a minute in between 
    querying the comments for the 1st batch, 2nd batch, and 3rd batch (to avoid going over asyncpraw's api limit)
    '''
    post_batches = []
    if len(posts) >= 200 and len(posts) <= 300:
        mid = len(posts) // 2
        post_batches.append(posts[:mid])
        post_batches.append(posts[mid:])
    elif len(posts) > 300:
        third = len(posts) // 3
        post_batches.append(posts[:third])
        post_batches.append(posts[third:2*third])
        post_batches.append(posts[2*third:])

    # Fetch comments for all posts
    posts_list = []
    if len(post_batches) > 0:
        for post_batch in post_batches:
            posts_with_comments = await asyncio.gather(*(fetch_post_data(post) for post in post_batch))
            posts_with_comments = [post for post in posts_with_comments if post is not None]
            posts_list.extend(posts_with_comments)
            print("got the comments for ", len(posts_with_comments), "/", len(post_batch), " of the posts") 
            time.sleep(65)  # Wait for 65 seconds to prevent exceeding praw's api limit 
    else:
        posts_list = await asyncio.gather(*(fetch_post_data(post) for post in posts))

    # Calculate total words across all content
    total_words = 0
    for post in posts_list:
        total_words += len(post.title.split()) + len(post.description.split()) + sum([len(comment.split()) for comment in post.comments])

    # Perform various analyses
    posts_grouped_by_date = group_posts_by_date(posts_list)
    comment_and_score_pairs_grouped_by_date = dict()
    for date, posts in posts_grouped_by_date.items():
        # very unlikely that 2 comments mentioning an entity will be EXACTLY the same 
        comment_and_score_pairs = dict() # across all posts in this date 
        for post in posts:
            for i in range(len(post.comments)):
                comment = post.comments[i]
                comment_score = post.comment_scores[i]
                comment_and_score_pairs[comment] = comment_score 
        comment_and_score_pairs_grouped_by_date[date] = comment_and_score_pairs
    readability_metrics = get_readability_metrics(posts_list)
    top_n_grams, top_named_entities, top_named_entities_embeddings, top_named_entities_wordcloud = await get_subreddit_analysis(posts_grouped_by_date, comment_and_score_pairs_grouped_by_date)
    
    # Compile final analysis
    analysis = SubredditAnalysis(
        timestamp = int(time.time()),
        num_words = total_words,
        subreddit = subreddit_query.name,
        top_n_grams = top_n_grams,
        top_named_entities = top_named_entities,
        top_named_entities_embeddings = top_named_entities_embeddings,
        top_named_entities_wordcloud = top_named_entities_wordcloud,
        readability_metrics =  readability_metrics,   
    )

    return analysis
