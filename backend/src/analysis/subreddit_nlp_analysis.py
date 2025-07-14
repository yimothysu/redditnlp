"""Module for performing comprehensive NLP analysis on subreddit content.
Includes functionality for post collection, text analysis, and feature extraction."""

import time, asyncio, asyncpraw, os, queue  # type: ignore
from typing import List
from dotenv import load_dotenv # type: ignore
load_dotenv()
from datetime import datetime

from src.data_fetchers.reddit_post_fetcher import ( fetch_post_data, Comment )
from src.utils.subreddit_classes import ( SubredditQuery, SubredditAnalysis, RedditPost )
from src.analysis.features.named_entities import ( get_top_entities )
from src.analysis.features.topics import ( get_topics )
from src.analysis.features.n_grams import ( get_top_ngrams )
from src.analysis.features.readability import ( get_readability_metrics )
from src.analysis.features.word_embeddings import ( get_2d_embeddings )
from src.analysis.features.word_cloud import ( generate_word_cloud )
from src.analysis.coreference_resolution import ( resolve_pronouns_for_post )

# Configuration maps for different time filters
TIME_FILTER_TO_INITIAL_POST_QUERY_LIMIT = {'all': 500, 'year': 300, 'week': 200}
TIME_FILTER_TO_POST_LIMIT = {'all': 150, 'year': 100, 'week': 50}
TIME_FILTER_TO_DATE_FORMAT = {'all': '%y', 'year': '%m-%y', 'week': '%m-%d'}

config = {
    "time_filter": "week",
    "max_post_len": 1000000, # in characters 
    "subreddit": ""
}

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


async def get_subreddit_analysis(posts: list[RedditPost]):
    """Perform NLP analysis on grouped posts.
    
    Args:
        posts: List of RedditPost objects 
    Returns:
        tuple: (top n-grams, top named entities)
    """
    # post_content_grouped_by_date = dict() 
    # post_title_and_description_grouped_by_date = dict() 
    # for date, posts in posts_grouped_by_date.items():
    #     post_content = '\n'.join([get_flattened_text_for_post(post, include_comments=True) for post in posts])
    #     posts_title_and_description = [get_flattened_text_for_post(post, include_comments=False) for post in posts]
    #     # Can write post_content to a text file so I can inspect it 
    #     with open('post_content.txt', 'w', encoding='utf-8') as f:
    #         f.write(post_content)
    #     post_content_grouped_by_date[date] = post_content
    #     post_title_and_description_grouped_by_date[date] = posts_title_and_description

    #top_n_grams = get_top_ngrams(post_content_grouped_by_date)
    #top_named_entities = await get_top_entities(config['subreddit'], config['time_filter'], post_content_grouped_by_date, posts_grouped_by_date, comment_and_score_pairs_grouped_by_date)
    # top_named_entities = dict() 
    # topics = dict() 
    # for date, posts in post_title_and_description_grouped_by_date.items():
    #     topics[date] = get_topics(config['subreddit'], posts)
    # # Generate embeddings and word cloud for named entities
    # entity_set = set()
    # for _, entities in top_named_entities.items():
    #     entity_names = [entity.name for entity in entities]
    #     for entity_name in entity_names: entity_set.add(entity_name)
    # top_named_entities_embeddings = get_2d_embeddings(list(entity_set))
    # top_named_entities_wordcloud = generate_word_cloud(top_named_entities)
    # readability_metrics = get_readability_metrics(posts)
    # return top_n_grams, topics, top_named_entities, top_named_entities_embeddings, top_named_entities_wordcloud, readability_metrics


    top_n_grams = get_top_ngrams(posts)
    topics = get_topics(config['subreddit'], posts) # type dict[str, list[str]]
    top_named_entities = get_top_entities(config['subreddit'], config['time_filter'], posts)
    top_named_entities_embeddings = dict() 
    top_named_entities_wordcloud = ""
    readability_metrics = get_readability_metrics(posts)
    return top_n_grams, topics, top_named_entities, top_named_entities_embeddings, top_named_entities_wordcloud, readability_metrics

     
'''
    total words = length of the comment and length of comment's replies 
'''
def calculate_total_words_of_comment(comment: Comment):
    # comment is of type Comment (has the fields 'text' and 'replies')
    total_words_of_comment = len(comment.text.split()) 
    comment_replies = comment.replies 
    for reply in comment_replies:
        total_words_of_comment += calculate_total_words_of_comment(reply)
    return total_words_of_comment


def print_comments(comments: List[Comment], indent: int = 0):
    for comment in comments:
        if comment.text:
            print(" " * indent + comment.text)
        if comment.replies:
            print_comments(comment.replies, indent + 4)


def print_post_tree(post):
    print('title: ', post.title)
    print('description: ', post.description)
    print_comments(post.top_level_comments)

async def perform_subreddit_analysis(subreddit_query: SubredditQuery):
    """Main function to perform comprehensive analysis on a subreddit.
    
    Fetches posts, analyzes content, and generates various NLP metrics.
    
    Args:
        subreddit_query: SubredditQuery object containing analysis parameters
    Returns:
        SubredditAnalysis: Complete analysis results
    """
    config['subreddit'] = subreddit_query.name
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
    )] # Object type = Submission 

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
    posts_list = [] # Object type = List[RedditPost]
    if len(post_batches) > 0:
        for post_batch in post_batches:
            posts_with_comments = await asyncio.gather(*(fetch_post_data(post) for post in post_batch))
            posts_with_comments = [post for post in posts_with_comments if post is not None]
            posts_list.extend(posts_with_comments)
            print("got the comments for ", len(posts_with_comments), "/", len(post_batch), " of the posts") 
            time.sleep(65)  # Wait for 65 seconds to prevent exceeding praw's api limit 
    else:
        posts_list = await asyncio.gather(*(fetch_post_data(post) for post in posts))
    posts_list = [post for post in posts_list if post is not None and isinstance(post, RedditPost)]
    # Calculate total words across all content
    total_words = 0
    for post in posts_list:
        total_words += len(post.title.split()) + len(post.description.split()) + sum([calculate_total_words_of_comment(top_level_comment) for top_level_comment in post.top_level_comments])
    print('total words: ', total_words)

    # Perform coreference resolution on all the posts text in posts_list 
    # for i in range(len(posts_list)):
    #     start_coref_res = time.time() 
    #     post = posts_list[i]
    #     posts_list[i] = resolve_pronouns_for_post(post) 
    #     end_coref_res = time.time() 
    #     print("coref resolution for post " + str(i) + " took: ", end_coref_res - start_coref_res)

    # Perform various analyses
    # posts_grouped_by_date = group_posts_by_date(posts_list)
    # comment_and_score_pairs_grouped_by_date = dict()
    # for date, posts in posts_grouped_by_date.items():
    #     # very unlikely that 2 comments mentioning an entity will be EXACTLY the same 
    #     comment_and_score_pairs = dict() # across all posts in this date 
    #     for post in posts:
    #         unvisited = queue.Queue()
    #         for top_level_comment in post.top_level_comments:
    #             unvisited.put(top_level_comment)
    #         while not unvisited.empty():
    #             comment = unvisited.get()
    #             comment_and_score_pairs[comment.text] = comment.score
    #             replies = comment.replies
    #             for reply in replies:
    #                 unvisited.put(reply)
    #     comment_and_score_pairs_grouped_by_date[date] = comment_and_score_pairs
    # readability_metrics = get_readability_metrics(posts_list)
    top_n_grams, topics, top_named_entities, top_named_entities_embeddings, top_named_entities_wordcloud, readability_metrics = await get_subreddit_analysis(posts_list)
    
    # Compile final analysis
    analysis = SubredditAnalysis(
        timestamp = int(time.time()),
        num_words = total_words,
        subreddit = subreddit_query.name,
        top_n_grams = top_n_grams,
        topics = topics, 
        top_named_entities = top_named_entities,
        top_named_entities_embeddings = top_named_entities_embeddings,
        top_named_entities_wordcloud = top_named_entities_wordcloud,
        readability_metrics =  readability_metrics,   
    )

    return analysis
