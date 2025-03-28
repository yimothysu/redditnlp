# gets all the top 20 subreddit analyses for a specific time filter and stores it in 
# the db 

import requests
from db import week, month, year, all_time 

time_filter = "week" # CHANGE IF YOU WANT A DIFFERENT TIME FILTER 
BASE_URL = "http://localhost:8000/analysis"
f = open("20_subreddits.txt", "r", encoding="utf-8")
subreddits = []
for line in f:
    subreddits.append(line.strip())  

def fetch_subreddit_data(subreddit):
    subreddit_query = {
        "name": subreddit,
        "time_filter": time_filter, 
        "sort_by": "top" 
    }

    response = requests.post(BASE_URL, json=subreddit_query)
    if response.status_code == 200:
        print("Response:", response.json())
        if time_filter == "week": week.insert_one(response.json())
        elif time_filter == "month": month.insert_one(response.json())
        elif time_filter == "year": year.insert_one(response.json())
        if time_filter == "all_time": all_time.insert_one(response.json())
    else:
        print("Error:", response.status_code, response.text)


if __name__ == "__main__":
    for subreddit in subreddits:
        fetch_subreddit_data(subreddit)
