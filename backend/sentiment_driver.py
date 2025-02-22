# This file is meant for sentiment analysis testing

import json

from subreddit_nlp_analysis import preprocess_text_for_sentiment, get_sentiment

with open("posts.txt") as f:
    posts = json.load(f)

sentiment_res = []

for post in posts:
    post_content = post['title'] + ' '.join(post['comments'])
    filtered_post_content = preprocess_text_for_sentiment(post_content)
    sentiment = get_sentiment(filtered_post_content)
    
    sentiment_res.append({
        "title": post['title'],
        "sentiment": sentiment
    })
    
for result in sentiment_res:
    print(result)

