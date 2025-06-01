import os
import requests
from dotenv import load_dotenv

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

def summarize_comments(entity, subreddit, comments):
    # Join comments into one prompt
    prompt = (
    "You are an analyst summarizing Reddit sentiment about " + entity + " from " + subreddit + "." 
    + "Here are the comments:\n\n" 
    + "\n".join(f"- {c}" for c in comments)
    + "Please write a short paragraph that summarizes:\n"
    + "- The overall sentiment (very positive, positive leaning, very negative, negative leaning, neutral, or mixed)\n"
    + "- Common positives mentioned by the users\n"
    + "- Common negatives mentioned by the users\n"
    + "Format the summary like this:\n" 
    + "Overall, the sentiment toward [ENTITY] on r/[SUBREDDIT] is [SENTIMENT].\n" 
    + "Users commonly mention [BULLET POINT THEMES].\n" 
    + "Some users also express [MINORITY OPINION], though others [COUNTERPOINT].\n"
    )
    print(prompt)
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
        return summary 
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        print("Full response content:")
        return "error"

print("\nSummary of subreddit sentiment about Tesla:\n")
print(summarize_comments("Tesla", "r/technology", comments))