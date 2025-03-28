import os

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from dotenv import load_dotenv

load_dotenv()

uri = os.environ["MONGODB_CONNECTION_STRING"]
client = MongoClient(uri, server_api=ServerApi("1"))
db = client["20_subreddits_analyses"]  
week = db["week_analyses"]  
month = db["month_analyses"]  
year = db["year_analyses"] 
all_time = db["all_time_analyses"] 