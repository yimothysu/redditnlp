import matplotlib.pyplot as plt
import datetime

def plot_post_distribution(subreddit, time_filter, posts_list):
    if time_filter not in ['year', 'month', 'week']:
        raise Exception(time_filter, ' is not a valid time filter')
    
    date_format = ''
    if time_filter == 'year': date_format = '%m-%y' # slicing a year by months 
    else: date_format = '%d-%m' # slicing a month and week by days 

    post_dates = [datetime.datetime.fromtimestamp(reddit_post.created_utc).strftime(date_format) for reddit_post in posts_list]

    date_to_num_posts = dict()
    for post_date in post_dates:
        if post_date not in date_to_num_posts: date_to_num_posts[post_date] = 1
        else: date_to_num_posts[post_date] += 1
    
    # sort date_to_num_posts by date
    dates = list(date_to_num_posts.keys())
    sorted_months = sorted(dates, key=lambda x: datetime.datetime.strptime(x, date_format))
    sorted_date_to_num_posts = {i: date_to_num_posts[i] for i in sorted_months}

    # plot it
    dates = sorted_date_to_num_posts.keys()
    num_posts = sorted_date_to_num_posts.values()
    plt.bar(dates, num_posts)
    title = 'Post Distribution for PRAW Query of ' + 'r/' + subreddit + ' past ' + time_filter + ' Top Posts'
    plt.title(title)
    if time_filter == 'year': plt.xlabel('Months')
    else: plt.xlabel('Days')
    plt.ylabel('# Posts')
    plt.show()
    