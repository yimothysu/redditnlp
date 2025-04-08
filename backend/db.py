import os

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from subreddit_classes import (
    SubredditQuery,
)

from dotenv import load_dotenv

load_dotenv()

uri = os.environ["MONGODB_CONNECTION_STRING"]
client = MongoClient(uri, server_api=ServerApi("1"))
db = client["20_subreddits_analyses"]  
week = db["week_analyses"]  
month = db["month_analyses"]  
year = db["year_analyses"] 
all = db["all_time_analyses"]

# defining 'subreddit' as the key for each collection 
week.create_index("subreddit", unique=True)
month.create_index("subreddit", unique=True)
year.create_index("subreddit", unique=True)
all.create_index("subreddit", unique=True)

COLLECTIONS = {
    "week": week,
    "month": month,
    "year": year,
    "all": all,
}

def fetch_subreddit_analysis(subreddit_query: SubredditQuery):
    collection = COLLECTIONS[subreddit_query.time_filter]
    return collection.find_one({"subreddit": subreddit_query.name})
