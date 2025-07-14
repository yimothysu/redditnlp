import os
import requests
import math 
import time 
import random
from dotenv import load_dotenv
from src.utils.subreddit_classes import ( RedditPost )

# Load .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# Expanded sample comments about Tesla
comments = [
    "I love my Tesla, it's smooth and fast.",
    "The repair process took forever, super frustrating.",
    "Autopilot saved me from an accident!",
    "Elon Musk is too involved in Twitter drama.",
    "The price went up again... ugh.",
    "Charging infrastructure is better than other EVs.",
    "Software updates are cool but buggy sometimes.",
    "Wish there was more customer support.",
    "Tesla is leading the EV market, no doubt.",
    "The Model 3 is the best car I've ever owned.",
    "I don't trust their build quality at all.",
    "Why can't they include Apple CarPlay already?",
    "The delivery experience was a nightmare.",
    "Customer service ghosted me when I had issues.",
    "Battery degradation is minimal after 3 years.",
    "FSD is still not truly autonomous.",
    "It's amazing how well it handles in snow.",
    "My insurance premium went up after switching to a Tesla.",
    "Feels like driving a spaceship, in a good way.",
    "Can't go back to gas after this experience."
]

posts = [
    RedditPost(title="Swampy memes could never", 
         description="", 
         idx=1, 
         url="post_1_url"),

    RedditPost(title="seniors/recent grads question",
         description="what is one thing about ur time at UF throughout all the advising and courses you wish you knew/knew sooner?"
                     "an opportunity or resource a specific class or instructor connections to research, internships, shadowing, etc"
                     "anything!\n",
         idx=2, 
         url="post_2_url"),

    RedditPost(title="[HELP] I just transferred to UF and owe over $4k for summer — didn’t realize I wouldn’t get financial aid until fall",
         description="Hi all, I’m in a tough spot and could really use some advice."
                     "I just transferred to the University of Florida from Santa Fe College. I recently found out that I owe over $4,000 for the summer semester — because my financial aid for the year had already been used while I was at Santa Fe. I didn’t realize that meant I wouldn’t get any aid for summer at UF."
                     "I do have financial aid lined up for the fall, but that doesn’t help me with the immediate summer bill. Right now, I have no money and no way to cover the balance."
                     "Is there any way to defer the payment, set up a payment plan, or somehow avoid being dropped from classes? Should I reach out to the bursar’s office, financial aid, or someone else?"
                     "If anyone has dealt with something like this — especially as a transfer student — I’d really appreciate hearing how you handled it. I’m just trying to stay enrolled and get through the summer without everything falling apart."
                     "Thanks in advance."
                     "Edit: I have terrible credit and no co-signers and I’m summer B\n",
         idx=3,
         url="post_3_url"),
    
    RedditPost(title="Butler plaza?",
         description="What’s going on at butler plaza? There’s like 50 cop cars blocking everything off",
         idx=4,
         url="post_4_url"),

    RedditPost(title="State Employees Will Receive an Extra Day Off for July 4th Weekend and a 2% raise?",
         description="Did any university employees receive this? From what I understand, no."
                     "“In anticipation of America’s upcoming semiquincentennial celebration in 2026, I am giving state employees a longer weekend this year to celebrate,” said Governor Ron DeSantis. “I hope our state employees use this additional time off to enjoy Florida’s"
                     "Freedom Summer with their loved ones and reflect on the importance and meaning of our nation’s founding.” The governor is also approving a 2% raise—funded through the state’s annual budget—for state employees, which will go into effect July 1, 2025.\n",
         idx=5,
         url="post_5_url"),

    RedditPost(title="Is UF insurance plan worth it",
         description="As the title says. I currently have very limited options and am looking into UFs insurance for the upcoming school year. I’ve heard not so great things about it and am not sure if it’ll be worth it considering, I’m looking for dental and vision " 
                     "coverage too. If anyone has experience with it, any input would be greatly appreciated.",
         idx=6,
         url="post_6_url"),
    
    RedditPost(title="Hey! Don’t mean to sh*t post but I’m doing an interesting survey on aerospace education.",
         description="Please give me privilege of 2 minutes to gauge how you feel about the aerospace industry and your general knowledge about it. *Not a quiz"
                     "https://docs.google.com/forms/d/e/1FAIpQLSfaxx4b6bbBhLZWg7YNXX_HK5wvlv0CXkmN3IRoczYhmTwYCA/viewform?usp=header",
         idx=7,
         url="post_7_url")
]


def summarize_comments(entity, subreddit, comments, num_comments_sample=75, try_again_if_too_many_requests=True):
    num_comments_sample = min(num_comments_sample, len(list(comments)))
    comments_sample = random.sample(list(comments), num_comments_sample)
    # Join comments into one prompt
    prompt = (
    "You are an analyst summarizing Reddit sentiment about " + entity + " from " + subreddit + "." 
    + "Here are the comments:\n\n" 
    + "\n".join(f"- {c}" for c in comments_sample)
    + "Please write a short paragraph that summarizes:\n"
    + "- The overall sentiment (very positive, positive leaning, very negative, negative leaning, neutral, or mixed)\n"
    + "- Common positives mentioned by the users\n"
    + "- Common negatives mentioned by the users\n"
    + "Format the summary like this:\n" 
    + "[SENTIMENT - VERY POSITIVE/MOSTLY POSITIVE/VERY NEGATIVE/MOSTLY NEGATIVE/NEUTRAL/CONTROVERSIAL].\n" 
    + "[SUMMARY]"
    )

    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    # Send the request
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        summary = response.json()["choices"][0]["message"]["content"]
        return summary 
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        if e.response is not None and e.response.status_code == 429:
            if try_again_if_too_many_requests:
                print("Too many requests! lets wait 2 minutes and try again. if its still unsuccessful, just give up")     
                time.sleep(120)
                return summarize_comments(entity, subreddit, comments, try_again_if_too_many_requests=False)
            else:
                print("2nd time trying because of too many requests. giving up now")
                return ""

        elif e.response is not None and e.response.status_code == 413:
            print("Payload too large!")
            # Retry but reduce # of comments by 10 
            time.sleep(30) # wait a bit before trying again
            if num_comments_sample - 10 > 0: 
                return summarize_comments(entity, subreddit, comments, num_comments_sample=num_comments_sample-10)
        return ""


ufl_topics = [
    'housing',
    'academics',
    'course selection',
    'transfer students',
    'uf resources',
    'job search',
    'student life',
    'uf employee benefits',
    'uf admissions',
    'major and course selection',
    'housing and roommates',
    'transferring to uf',
    'uf academics and advising',
    'off-campus opportunities',
    'online and distance learning',
    'time management',
    'research',
    'class scheduling',
    'travel',
    'academic concerns',
    'easy classes',
    'online learning',
    'transfer',
    'engineering',
    'academic advice',
    'financial aid',
    'housing',
    'roommate search',
    'research opportunities',
    'study tips',
    'cheating',
    'college life',
    'uf academics',
    'uf classes',
    'one.uf issues',
    'summer opportunities',
    'uf spirit',
    'financial aid',
    'major choice',
    'uf student senate',
    'apartment hunting',
    'apartment reviews',
    'gainesville, fl apartments',
    'renting nightmares',
]


def is_list_of_reddit_posts(obj):
    # print('type(obj): ', type(obj))
    # for item in obj:
    #     print('type(item): ', type(item))
    #     print('item: ', item)
    # if isinstance(obj, list) and all(isinstance(item, RedditPost) for item in obj):
    #     raise Exception("is list of reddit posts")
    return (
        isinstance(obj, list)
        and all(isinstance(item, RedditPost) for item in obj)
    )


def combine_similar_topics(topic_to_posts: dict[str, list[RedditPost]], subreddit):
    # first verify that each value in topic_to_posts is actually a list of RedditPost objects. If it is, that means 
    # something changed in topic_to_posts after querying groq 
    for topic, posts in topic_to_posts.items():
        if not is_list_of_reddit_posts(posts):
            raise ValueError("Invalid input into combine_similar_topics: expected a dict where the value in each key-value pair is a list[RedditPost].")
        
    topics = list(topic_to_posts.keys()) 
    prompt = (
        "You are an analyst given a list of topics extracted from the subreddit " + subreddit + "." 
        + "Here is the list of the topics:\n\n" 
        + "\n".join(topic for topic in topics)
        + "Here are some examples to demonstrate how I want you to combine topics: \n"
        + "- you can combine 'academics', 'academic advice', 'uf academics' into just one topic: academics \n"
        + "- you can combine 'course selection', 'uf classes', 'major and course selection', 'easy classes', 'major choice' into just one topic: major and course selection\n"
        + "- you can combine 'housing', housing and roommates', 'roommate search', 'apartment hunting', 'apartment reviews', 'gainesville, fl apartments', 'renting nightmares' into just one topic: housing and roommate search\n"
        + "- you can combine 'time management', 'study tips', 'academic advice' into just one topic: academic advice & tips\n"
        + "- you can combine 'transfer students', 'transferring to uf', 'transfer' into just one topic: transfer students\n"
        + "- you can combine 'research opportunities', 'job search', 'off-campus opportunities', 'summer opportunities' into just one topic: research/job opportunities\n"
        + "Format your response like this: \n"
        + "COMBINED_TOPIC \ TOPIC_1 == TOPIC_2 == TOPIC_3 etc.\n"
        + "COMBINED_TOPIC \ TOPIC_1 == TOPIC_2 == TOPIC_3 etc.\n"
        + "COMBINED_TOPIC \ TOPIC_1 == TOPIC_2 == TOPIC_3 etc.\n"
        )

    # Prepare request data
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    # Send the request
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        combined_topics = response.json()["choices"][0]["message"]["content"]
        #print(f"groq output for combining:\n{combined_topics}")
        # Parse the result 
        combined_topic_to_posts = dict() 
        lines = combined_topics.splitlines()
        for line in lines:
            # Each line has the format: COMBINED_TOPIC \ TOPIC_1 == TOPIC_2 == TOPIC_3 etc.
            try:
                combined_topic, topics = line.split(" \ ")
                combined_topic = combined_topic.strip() 
                topics = topics.strip().split(" == ")
                combined_topic_to_posts[combined_topic] = []
                for topic in topics:
                    if topic in topic_to_posts and is_list_of_reddit_posts(topic_to_posts[topic]):
                        combined_topic_to_posts[combined_topic] += topic_to_posts[topic]
                    else:
                        print("topic ISNT in topic_to_posts")
            except:
                continue
        return combined_topic_to_posts 
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        print("Full response content:")
        return dict() 
    
    
def remove_description_from_reddit_post_object(post: RedditPost):
    post.description = ""
    post.top_level_comments = []
    return post 


# Returns a dict called topics where 
#   key (type string) = topic 
#   value (type list[string]) = list of post urls 
def summarize_posts(subreddit: str, posts: list[RedditPost], check_prompt_length=True, split_in_case_error=True, try_again_if_too_many_requests=True):
    # each post has the format 
    #   Title: [TITLE]
    #   Description: [DESCRIPTION]
    prompt = (
    "You are an analyst summarizing the topics of these " + str(len(posts)) + " reddit posts from the subreddit " + subreddit + "." 
    + "Here is the list of the post titles and their descriptions:\n\n" 
    + "\n".join(f"\nPost {p.idx}:\nTITLE: {p.title if p.title is not None else ""}\nDESCRIPTION:{p.description if p.description is not None else ""}\n" for p in posts)
    + "Please extract the topics of the posts and assign which posts correspond to which topics. A post can be in more than one topic category\n"
    # + "Be as concise as possible, and make the response optimal for skimming and brief glancing."
    + "Try your best to combine topics that any 2 posts have in common."
    + "Format the summary like this: \n"
    + "TOPIC_1 / POST_IDX etc.\n"
    + "TOPIC_2 / POST_IDX etc.\n"
    + "TOPIC_3 / POST_IDX etc.\n"
    + "Your response should ONLY contain the summary. Please do not append any text before and after the summary."
    + "The post indices following each topic should be seperated only be a space. For example '1 2 3' is valid. '1, 2, 3' is not valid."
    )
    #print(f"prompt: \n{prompt}")
    # problem is past here 
    if len(prompt) > 15000 and check_prompt_length:
        num_subprompts = math.ceil(len(prompt) / 10000)
        k, m = divmod(len(posts), num_subprompts)
        posts_groups = [posts[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_subprompts)]
        aggregated_topics = dict() 
        for posts_group in posts_groups:
            time.sleep(60)
            posts_group_topics = summarize_posts(subreddit, posts_group, check_prompt_length=False)
            # add posts_group_topics into aggregated topics 
            for topic, posts_for_topic in posts_group_topics.items():
                if topic not in aggregated_topics:
                    aggregated_topics[topic] = posts_for_topic
                else:
                    aggregated_topics[topic] = aggregated_topics[topic] + posts_for_topic
                # combine similar topics 
        # problem before this 
        aggregated_topics = combine_similar_topics(aggregated_topics, subreddit)
        
        # Remove description field from RedditPost objects to save mongodb memory 
        for topic, posts_for_topic in aggregated_topics.items():
            new_posts = []
            for post in posts_for_topic:
                new_post = remove_description_from_reddit_post_object(post)
                new_posts.append(new_post) 
            aggregated_topics[topic] = new_posts 
        
        return aggregated_topics 
    # Prepare request data
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    # Send the request
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        summary = response.json()["choices"][0]["message"]["content"]
        #print(f"groq output for summarizing:\n{summary}")
        # Parse summary into a dict where key = topic, value = list of post urls 
        post_idx_to_post_object = dict() 
        for post in posts:
            post_idx_to_post_object[post.idx] = post
        topics = dict() 
        lines = summary.splitlines()
        for line in lines:
            # Each line has the format: TOPIC_1 / POST_IDX etc.
            try:
                topic, post_indices = line.split(" / ")
                topic = topic.strip() 
                post_indices = list(map(int, post_indices.strip().split()))
                post_objects = [post_idx_to_post_object[post_idx] for post_idx in post_indices]
                topics[topic] = post_objects
            except:
                continue
        # Remove description field from RedditPost objects to save mongodb memory 
        # print topics to a txt file 
        # with open("topics.txt", "w", encoding="utf-8") as f:
        #     print(topics, file=f)
        for topic, posts_for_topic in topics.items():
            # sometimes groq surrounds the topic in ** (Ex: **science**). Get rid of the ** characters 
            if topic.startswith("**"): topic = topic[2:]
            if topic.endswith("**"): topic = topic[:-2]

            new_posts = []
            for post in posts_for_topic:
                new_post = remove_description_from_reddit_post_object(post)
                new_posts.append(new_post) 
            topics[topic] = new_posts 

        return topics 

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        if e.response is not None and e.response.status_code == 429:
            if try_again_if_too_many_requests:
                print("Too many requests! lets wait 2 minutes and try again. if its still unsuccessful, just give up")     
                time.sleep(120)
                return summarize_posts(subreddit, posts, try_again_if_too_many_requests=False)
            else:
                print("2nd time trying because of too many requests. giving up now")

        elif e.response is not None and e.response.status_code == 413:
            print("Payload too large!")
            if split_in_case_error:
                # Retry but split posts in half 
                mid = len(posts) // 2
                first_half = posts[:mid]
                second_half = posts[mid:]
                time.sleep(60)
                first_half_topics = summarize_posts(subreddit, first_half, check_prompt_length=False, split_in_case_error=False)
                time.sleep(60)
                second_half_topics = summarize_posts(subreddit, second_half, check_prompt_length=False, split_in_case_error=False)
                
                # combine first_half_topics and second_half_topics into one dict 
                combined_topics = dict() 
                for topic, posts_for_topic in first_half_topics.items():
                    if topic not in combined_topics:
                        combined_topics[topic] = []
                    combined_topics[topic] += posts_for_topic
                for topic, posts_for_topic in second_half_topics.items():
                    if topic not in combined_topics:
                        combined_topics[topic] = []
                    combined_topics[topic] += posts_for_topic

                return combined_topics
            
        return dict() 

# print("Summary of r/ufl posts: \n")
# summarize_posts("r/ufl", posts)
