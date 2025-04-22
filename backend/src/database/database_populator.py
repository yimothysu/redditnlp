"""Script to populate MongoDB collections with subreddit analysis data.
Performs NLP analysis on top posts from specified subreddits and stores results.
"""

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
    "AskHistorians",
    "unpopularopinion",
    "philosophy",
    "mentalhealth",
    "teenagers",
    "AskMen",
    "AskWomen",
    "personalfinance",
    "changemyview",
    "LateStageCapitalism",
    "UpliftingNews",
    "ApplyingToCollege",
    "conspiracytheories",
    "linguistics",
    "lgbt",
    "DrugNerds",
    "Bitcoin",
    "Futurology",
    "findapath",
    "Parenting",
    "4chan",
    "TheRedPill",
    "TheBluePill",
    "SkincareAddiction",
    "Conservative",
    "Libertarian",
    "antiwork",
    "Christianity",
    "uofm",
    "Harvard",
    "stanford", 
    "mit",
    "berkeley",
    "UPenn",
    "yale",
    "columbia",
    "princeton",
    "ufl",
]


async def fetch_subreddit_data(subreddit, time_filter):
    """Fetch and store NLP analysis data for a specific subreddit.
    
    Args:
        subreddit (str): Name of the subreddit to analyze
        time_filter (str): Time period to analyze ('week', 'year', 'all')
    """
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
    if len(sys.argv) != 3:
        print("Error: must pass in time filter argument and start from index")
        sys.exit(1)
    
    time_filter = sys.argv[1]
    start_from_idx = sys.argv[2]
    if time_filter not in ['week', 'year', 'all']:
        print("Error: time filter argument must be 'week', 'year', or 'all'")
        sys.exit(1)
    
    try:
        if int(start_from_idx) < 0 or int(start_from_idx) >= len(subreddits):
            print("Error: not a valid start from index argument. Must be between 0 and ", len(subreddits), " inclusive")
    except: 
        print("Error: start from index argument must be an integer")
        sys.exit(1)
    
    start_from_idx = int(start_from_idx)
    print('start_from_idx: ', start_from_idx)
    for i in range(start_from_idx, len(subreddits)):
        subreddit = subreddits[i]
        print(f"Working on subreddit {subreddit}")
        asyncio.run(fetch_subreddit_data(subreddit, time_filter))
