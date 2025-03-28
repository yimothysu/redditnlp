import os

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from dotenv import load_dotenv

load_dotenv()

uri = os.environ["MONGODB_CONNECTION_STRING"]
client = MongoClient(uri, server_api=ServerApi("1"))
