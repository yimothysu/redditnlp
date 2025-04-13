# Reddit Topic Extractor

This script fetches Reddit data from a specified subreddit and extracts topics using BERTopic, a topic modeling technique. It's based on the NLP analysis functionality from the existing codebase.

## Usage

Run the script with Python:

```bash
python reddit_topic_extractor.py
```

The script will prompt you for:
1. A subreddit name (without the r/ prefix)
2. A time filter (hour, day, week, month, year, or all)
3. The number of topics to extract

## Output

The script outputs:
1. Topics extracted from the subreddit posts and comments, organized by time slices
2. Each topic includes representative words and their weights
3. Results are also saved to a JSON file named `{subreddit}_{time_filter}_topics.json`

## Example

```
Enter a subreddit name (without r/): python

Time filter options:
1. hour
2. day
3. week (default)
4. month
5. year
6. all
Choose time filter (1-6, default is 3): 3

Enter number of topics to extract (default is 10): 5

Fetching data from r/python (time filter: week)...
Found posts in 4 time slices
Extracting topics...

Extracted Topics for r/python (time filter: week):

--- Time Slice: 04-20 ---
Top Topics:
Topic 0:
  - python: 0.0123
  - code: 0.0098
  - library: 0.0087
  - function: 0.0076
  - data: 0.0065

...

Results saved to python_week_topics.json
```