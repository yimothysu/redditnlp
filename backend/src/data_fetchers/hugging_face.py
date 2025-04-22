"""
Analyzes emotions in Reddit posts and comments using HuggingFace's emotion classification model.
Calculates weighted emotion distributions based on post/comment scores.
"""

from transformers import pipeline
import json 

# Load Reddit posts data
with open("posts.txt") as f:
    posts = json.load(f)

# Initialize emotion classifier
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", truncation=True)

# this model has 6 emotion categories 
# anger, disgust, fear, joy, neutral, sadness, surprise

for post in posts:
    # Combine title and description for analysis
    post_title_and_description = post['title'] + '. ' + post['description']
    emotion_scores = dict()
    
    # Analyze post content
    post_title_and_description_sentiment = classifier(post_title_and_description)
    emotion_scores[post_title_and_description_sentiment[0]['label']] = [(post_title_and_description_sentiment[0]['score'], post['score'])]
    print('post title: ', post['title'])
    
    # Analyze each comment
    for x in range(len(post['comments'])):
        comment = post['comments'][x]
        comment_score = post['comment_scores'][x]
        comment_sentiment = classifier(comment)
        if comment_sentiment[0]['label'] in emotion_scores: 
            emotion_scores[comment_sentiment[0]['label']].append((comment_sentiment[0]['score'], comment_score))
        else:
            emotion_scores[comment_sentiment[0]['label']] = [(comment_sentiment[0]['score'], comment_score)]
        
    for emotion, score_and_num_upvotes_tuples in emotion_scores.items():
        # conducting a weighted average 
        total_upvotes = 0
        total_weight = 0
        for score, num_upvotes in score_and_num_upvotes_tuples:
            total_upvotes += num_upvotes
            total_weight += (score * num_upvotes)
        weighted_average = total_weight / total_upvotes
        emotion_scores[emotion] = weighted_average * total_upvotes
   
    # conducting a weighted average to get distribution of emotions 
    total_weight = 0
    for emotion, weight in emotion_scores.items():
        total_weight += weight
    
    for emotion, weight in emotion_scores.items():
        percent = round((weight / total_weight) * 100)
        print(percent, '% ', emotion)
    print()