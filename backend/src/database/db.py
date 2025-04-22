"""MongoDB database configuration and utility functions.
Handles connection setup and provides access to collections storing subreddit analyses.
"""

import os

from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from src.utils.subreddit_classes import SubredditQuery

# Load MongoDB connection string from environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '../../config/.env')
load_dotenv(dotenv_path)

uri = os.environ["MONGODB_CONNECTION_STRING"]
client = MongoClient(uri, server_api=ServerApi("1"))
db = client["20_subreddits_analyses"]  

# Collections for different time periods of analysis
week = db["week_analyses"]  
year = db["year_analyses"] 
all = db["all_time_analyses"]

# defining 'subreddit' as the key for each collection 
week.create_index("subreddit", unique=True)
year.create_index("subreddit", unique=True)
all.create_index("subreddit", unique=True)

COLLECTIONS = {
    "week": week,
    "year": year,
    "all": all,
}


def fetch_subreddit_analysis(subreddit_query: SubredditQuery):
    """Retrieve analysis results for a specific subreddit query.
    
    Args:
        subreddit_query (SubredditQuery): Query parameters including subreddit name and time filter
        
    Returns:
        dict: Analysis results for the specified subreddit, or None if not found
    """
    collection = COLLECTIONS[subreddit_query.time_filter]
    return collection.find_one({"subreddit": subreddit_query.name})
