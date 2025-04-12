# gets all the top 20 subreddit analyses for a specific time filter and stores it in 
# the db 

from src.database.db import COLLECTIONS
import asyncio
import sys

from src.analysis.subreddit_nlp_analysis import (
    perform_subreddit_analysis,
)

from src.utils.subreddit_classes import (
    SubredditQuery,
)

subreddits = [
    "AskReddit",
    "politics",
    "AskOldPeople",
    "gaming",
    "science",
    "popculturechat",
    "worldnews",
    "technology",
    "100YearsAgo",
    "Feminists",
    "unpopularopinion",
    "philosophy",
    "mentalhealth",
    "teenagers",
    "AskMen",
    "AskWomen",
    "personalfinance",
    "changemyview",
    "LateStageCapitalism",
    "UpliftingNews"
]

async def fetch_subreddit_data(subreddit, time_filter):
    subreddit_query = SubredditQuery(
        name=subreddit,
        time_filter=time_filter,
        sort_by="top"
    )

    analysis = await perform_subreddit_analysis(subreddit_query)
    collection = COLLECTIONS[subreddit_query.time_filter]
    collection.update_one(
        {"id_": subreddit_query.name},
        {"$set": analysis.model_dump()},
        upsert=True # insert if no match
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: must pass in time filter argument")
        sys.exit(1)
    
    time_filter = sys.argv[1]
    if(time_filter not in ['week', 'year', 'all']):
        print("Error: time filter argument must be 'week', 'year', or 'all'")
        sys.exit(1)
    
    for subreddit in subreddits:
        print(f"Working on subreddit {subreddit}")
        asyncio.run(fetch_subreddit_data(subreddit, time_filter))
