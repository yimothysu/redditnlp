"""Script to populate MongoDB collections with subreddit analysis data.
Performs NLP analysis on top posts from specified subreddits and stores results.
"""

from src.database.db import COLLECTIONS
import asyncio
import sys
import bson 

from src.analysis.subreddit_nlp_analysis import (
    perform_subreddit_analysis,
)

from src.utils.subreddit_classes import (
    SubredditQuery,
)

subreddits = [
    "AskReddit",
    "technology",
    "politics",
    "gaming",
    "science",
    "popculturechat",
    "cscareerquestions",
    "stocks",
    "investing",
    "artificial",
    "AskHistorians",
    "unpopularopinion",
    "philosophy",
    "mentalhealth",
    "teenagers",
    "AskMen",
    "AskWomen",
    "personalfinance",
    "changemyview",
    "ApplyingToCollege",
    "conspiracytheories",
    "lgbt",
    "Bitcoin",
    "Futurology",
    "findapath",
    "Parenting",
    "4chan",
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
    # Serialize and measure size
    document = analysis.model_dump()
    document_size_bytes = len(bson.BSON.encode(document))
    document_size_mb = document_size_bytes / (1024 * 1024)
    print(f"Document size: {document_size_mb:.2f} MB")
    
    collection = COLLECTIONS[subreddit_query.time_filter]
    result = collection.update_one(
            {"id_": subreddit_query.name},
            {"$set": analysis.model_dump()},
            upsert=True # insert if no match
    )

    # Check if the collection was updated
    if result.matched_count > 0:
        if result.modified_count > 0:
            print("The document was updated.")
        else:
            print("The document was found but no changes were made (it may be the same).")
    else:
        print("No matching document was found, but a new one was inserted.")


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
