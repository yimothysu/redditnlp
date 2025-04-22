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

TIME_FILTER_TO_INITIAL_POST_QUERY_LIMIT = {'all': 1000, 'year': 1000, 'week': 200}
TIME_FILTER_TO_POST_LIMIT = {'all': 350, 'year': 200, 'week': 50}
TIME_FILTER_TO_NAMED_ENTITIES_LIMIT = {'all': 12, 'year': 10, 'week': 8}
TIME_FILTER_TO_DATE_FORMAT = {'all': '%y', 'year': '%m-%y', 'week': '%m-%d'}

config = {
    "time_filter": "week",
    "max_post_len": 1000000 # in characters 
}

async def get_post_title_and_description(post):
    try:
        flattened_post_text = str(post.title) + "\n" + str(post.selftext)
        upvotes = post.score
        return (flattened_post_text, upvotes)
    except Exception as e:
        print('could not get post title and description: ', e)

'''
 posts is a list of RedditPost objects
 function returns a sorted map by date where:
   key = date (Ex: 07/24)
   value = reddit posts with a timestamp inside [date]
'''
def group_posts_by_date(posts):
    posts_grouped_by_date = dict()
    date_format = TIME_FILTER_TO_DATE_FORMAT[config['time_filter']]
    for post in posts:
        post_date = datetime.fromtimestamp(post.created_utc).strftime(date_format)

        if post_date not in posts_grouped_by_date:
            posts_grouped_by_date[post_date] = []
        posts_grouped_by_date[post_date].append(post)

    # sort posts_grouped_by_date by date
    dates = list(posts_grouped_by_date.keys())
    sorted_months = sorted(dates, key=lambda x: datetime.strptime(x, date_format))
    posts_grouped_by_date = {i: posts_grouped_by_date[i] for i in sorted_months}
    return posts_grouped_by_date


async def get_subreddit_analysis(posts_grouped_by_date):
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
    top_named_entities = await get_top_entities(config['time_filter'], post_content_grouped_by_date, posts_grouped_by_date)
    return top_n_grams, top_named_entities


async def perform_subreddit_analysis(subreddit_query: SubredditQuery):
    config['time_filter'] = subreddit_query.time_filter

    reddit = asyncpraw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="reddit sampler",
    )

    initial_post_query_limit = TIME_FILTER_TO_INITIAL_POST_QUERY_LIMIT[subreddit_query.time_filter]

    subreddit_instance = await reddit.subreddit(subreddit_query.name)
    posts = [post async for post in subreddit_instance.top(
        limit=initial_post_query_limit,
        time_filter=subreddit_query.time_filter
    )]
    print("initially queried ", len(posts), posts)

    if(len(posts) > TIME_FILTER_TO_POST_LIMIT[subreddit_query.time_filter]):
        # Get the posts with the most amount of description text 
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
    print("# of posts after fetching comments: ", len(posts_list))

    total_words = 0
    for post in posts_list:
        total_words += len(post.title.split()) + len(post.description.split()) + sum([len(comment.split()) for comment in post.comments])
    print('total_words: ', total_words)

    posts_grouped_by_date = group_posts_by_date(posts_list)
    print('got posts from the dates: ', posts_grouped_by_date.keys())
    
    readability_metrics = get_readability_metrics(posts_list)
    print('Finished getting readability_metrics')
    print(readability_metrics)

    #plot_post_distribution(subreddit, time_filter, posts_grouped_by_date)
    top_n_grams, top_named_entities = await get_subreddit_analysis(posts_grouped_by_date)
    print(top_named_entities)

    entity_set= set()
    for _, entities in top_named_entities.items():
        entity_names = [entity.name for entity in entities]
        for entity_name in entity_names: entity_set.add(entity_name)
    top_named_entities_embeddings = get_2d_embeddings(list(entity_set))
    print(top_named_entities_embeddings)
    top_named_entities_wordcloud = generate_word_cloud(top_named_entities)
    
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
