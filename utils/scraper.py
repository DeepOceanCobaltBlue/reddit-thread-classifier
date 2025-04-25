"""
scraper.py

This module is used to download posts(called submissions) and their comments.
This is done by specifying the target subreddits and desired number of posts.
This module utilizes Python Reddit API Wrapper(PRAW).
flatten_comment(), flatten_submission(), used to strip undesired extras from 
downloaded submission. Leaving specified fields only and converting to dictionary.


Known limitations: Tends to only download ~ 1000 despite higher assigned value
of POSTS_PER_SUBREDDIT. 

Download file system setup 
C:\reddit data\filtered\
    ├── politics\
    │   ├── submissions\scraped.jsonl
    │   └── comments\scraped.jsonl

comments are written to file as they are downloaded to preserve data accumulation
in case of interruption.
"""

import praw
import json
import time
from pathlib import Path

# 🔐 Fill in your credentials
CLIENT_ID = "jakQDhlJIMGE5A0mb_lK6g"
CLIENT_SECRET = "lWPRK-1dDLMcIlz_4ZF8ixCWBsy8ZQ"
USER_AGENT = "web:SocComp:1.0 (by /u/Cobaltsixme)"

# 📂 Base output directory
BASE_OUT = Path(r"C:\reddit data\filtered")

# 🎯 Subreddits and target count
TARGET_SUBREDDITS = ['politics', 'askreddit']
POSTS_PER_SUBREDDIT = 1000

# 🛠 Initialize Reddit API
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

def flatten_comment(c, link_id, subreddit_name):
    return {
        "id": c.id,
        "parent_id": c.parent_id,
        "link_id": f"t3_{link_id}",
        "body": c.body,
        "score": c.score,
        "created_utc": c.created_utc,
        "subreddit": subreddit_name
    }

def flatten_submission(s):
    return {
        "id": s.id,
        "title": s.title,
        "selftext": s.selftext,
        "score": s.score,
        "url": s.url,
        "num_comments": s.num_comments,
        "created_utc": s.created_utc,
        "subreddit": s.subreddit.display_name
    }

def scrape_subreddit(subreddit_name, max_posts):
    print(f"\n🔍 Scraping r/{subreddit_name}...")
    sub_obj = reddit.subreddit(subreddit_name)

    # Paths
    sub_dir = BASE_OUT / subreddit_name / "submissions"
    com_dir = BASE_OUT / subreddit_name / "comments"
    sub_dir.mkdir(parents=True, exist_ok=True)
    com_dir.mkdir(parents=True, exist_ok=True)

    sub_path = sub_dir / "scraped.jsonl"
    com_path = com_dir / "scraped.jsonl"

    sub_count = 0
    with open(sub_path, "w", encoding="utf-8") as sub_out, \
         open(com_path, "w", encoding="utf-8") as com_out:

        for submission in sub_obj.new(limit=max_posts):
            # Write submission
            sub_data = flatten_submission(submission)
            sub_out.write(json.dumps(sub_data) + "\n")
            sub_count += 1
            print(f"[{sub_count}] Submission: {submission.id}")

            # Load comments
            try:
                submission.comments.replace_more(limit=0) # eliminate "load more" objects/pagination of comments
                for comment in submission.comments.list():
                    com_data = flatten_comment(comment, submission.id, subreddit_name)
                    com_out.write(json.dumps(com_data) + "\n")
            except Exception as e:
                print(f"⚠️ Failed to load comments for {submission.id}: {e}")

            # Be nice to Reddit's API
            time.sleep(0.2)

    print(f"\n✅ Done: {sub_count} submissions written for r/{subreddit_name}")

if __name__ == "__main__":
    for sub in TARGET_SUBREDDITS:
        scrape_subreddit(sub, POSTS_PER_SUBREDDIT)
