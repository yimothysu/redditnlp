"""Module for visualizing post distribution across time periods."""

import matplotlib.pyplot as plt

def plot_post_distribution(subreddit, time_filter, sorted_slice_to_posts):
    """Plot a bar chart showing post distribution over time.
    
    Args:
        subreddit: Name of the subreddit
        time_filter: Time period ('year', 'month', or 'week')
        sorted_slice_to_posts: Dictionary mapping dates to lists of posts
        
    Raises:
        Exception: If time_filter is invalid
    """
    if time_filter not in ['year', 'month', 'week']:
        raise Exception(time_filter, ' is not a valid time filter')

    dates = sorted_slice_to_posts.keys()
    num_posts = [len(posts_in_slice) for posts_in_slice in sorted_slice_to_posts.values()]
    plt.bar(dates, num_posts)
    title = 'Post Distribution for PRAW Query of ' + 'r/' + subreddit + ' past ' + time_filter + ' Top Posts'
    plt.title(title)
    if time_filter == 'year': plt.xlabel('Months')
    else: plt.xlabel('Days')
    plt.ylabel('# Posts')
    plt.show()
    