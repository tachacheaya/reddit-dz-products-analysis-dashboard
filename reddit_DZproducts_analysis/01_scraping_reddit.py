"""
    Code 01 : Data Collection from Reddit Subs
    
    - We are going to collect product reviews from Reddit subs
    
    -These subs are: r\AlgeriaRates & r\Algeria

    -Output: Saves the data collected in data\raw\product_reviews.csv
     """

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
import os
import json

#force the script to always run from the current folder (not parent)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# 1. define products keywords

KEYWORDS = {
    'food': [
        'rouiba', 'tchin-lait', 'hamoud', 'candia', 'ifri',
        'pasta', 'oil', 'milk', 'bread', 'flour', 'juice', 'coffee',
        'chocolate', 'honey', 'butter', 'rice', 'couscous', 'date',
        'yogurt', 'cheese', 'egg', 'meat', 'chicken', 'vegetable', 'fruit', 'bifa', 'bimo', 'kool','tazej'
    ],
    'electronics': [
        'condor', 'mobilis', 'djezzy', 'ooredoo', 'samsung', 'xiaomi',
        'phone', 'smartphone', 'computer', 'laptop', 'tablet', 'tv',
        'charger', 'battery', 'headphone', 'speaker', 'earphone',
        'screen', 'keyboard', 'mouse', 'printer', 'router'
    ],
    'appliances': [
        'condor', 'lg', 'samsung', 'whirlpool', 'hisense', 'sony',
        'fridge', 'refrigerator', 'washing machine', 'oven', 'microwave',
        'stove', 'cooker', 'heater', 'air conditioner', 'kettle', 'iron',
        'blender', 'mixer', 'toaster', 'dishwasher', 'dryer'
    ],
    'cosmetics': [
        'shampoo', 'cream', 'lotion', 'soap', 'perfume', 'makeup',
        'lipstick', 'foundation', 'mask', 'deodorant', 'toothpaste',
        'loreal', 'nivea', 'garnier', 'dove', 'skin care', 'hair care',
        'moisturizer', 'sunscreen', 'serum', 'conditioner', 'touché', 'natribifluor','emmanuelle jane', 'cosrx', 'burberry', 'felari'
    ]
}

ALL_KEYWORDS = [word for words in KEYWORDS.values() for word in words]


#2. create session with retry logic

def create_session():
    """create requests session with automatic retries"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,            #nbr of retries
        backoff_factor=2,   #wait 2,4, 8 sec between every retry
        status_forcelist=[429, 500, 502, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


# 3. helper functions

def is_product_review(text):
    """check if the post is about a product or not"""
    text = text.lower()
    for keyword in ALL_KEYWORDS:
        if keyword in text:
            return True
    return False


def get_category(text):
    """"shows which category a product belongs to"""
    text = text.lower()
    for category, words in KEYWORDS.items():
        for word in words:
            if word in text:
                return category
    return 'other'


def load_existing_reviews(filepath):
    """load existing reviews to avoid re-scraping from reddit"""
    if os.path.exists(filepath):
        try:
            existing_df = pd.read_csv(filepath)
            existing_ids = set(existing_df['id'].values)
            reviews_list = existing_df.to_dict('records')
            print(f"Loaded {len(existing_df)} existing reviews from {filepath}")
            return reviews_list, existing_ids
        except Exception as e:
            print(f"Could not load existing file: {e}")
            return [], set()
    return [], set()


def save_checkpoint(reviews, filepath):
    """save progress to csv"""
    temp_df = pd.DataFrame(reviews)
    temp_df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"  [Checkpoint] saved {len(reviews)} reviews")


# 4. main scraping function

def scrape_reddit_reviews(target=500, checkpoint_file='data/raw/product_reviews.csv'):

    # load existing reviews
    all_reviews, seen_ids = load_existing_reviews(checkpoint_file)

    # create a session with retries
    session = create_session()

    # subreddits to scrape
    subreddits = ['algeriarates', 'algeria', 'algerie', 'AskAlgeria']

    # sort methods
    sort_methods = ['new', 'hot', 'top']

    # track attempts and progress
    max_attempts = 5
    attempt = 0

    print("=" * 60)
    print("STEP 1: WEB SCRAPING - DATA COLLECTION (ROBUST VERSION)")
    print("=" * 60)
    print(f"Loaded {len(ALL_KEYWORDS)} product keywords")
    print(f"Categories: {list(KEYWORDS.keys())}")
    print(f"Target: {target} product reviews")
    print(f"Already collected: {len(all_reviews)}")
    print(f"Still needed: {max(0, target - len(all_reviews))}")
    print("")

    # continue scraping until the target = 500 is reached
    while len(all_reviews) < target and attempt < max_attempts:
        attempt += 1
        print(f"\n{'='*50}")
        print(f"SCRAPING ATTEMPT {attempt}/{max_attempts}")
        print(f"Collected: {len(all_reviews)}/{target}")
        print(f"{'='*50}")

        for subreddit in subreddits:
            if len(all_reviews) >= target:
                break

            print(f"\n[Subreddit] r/{subreddit}")

            for sort_method in sort_methods:
                if len(all_reviews) >= target:
                    break

                print(f"   Fetching {sort_method} posts....")

                # URL for reddit's JSON API
                url = f"https://www.reddit.com/r/{subreddit}/{sort_method}.json"
                after = None
                pages = 0
                max_pages = 50
                consecutive_errors = 0

                # progress bar for pages
                pbar = tqdm(total=max_pages, desc=f"    Pages ({sort_method})", leave=False)

                while len(all_reviews) < target and pages < max_pages and consecutive_errors < 3:
                    params = {'limit': 100}  # Get 100 posts per page
                    if after:
                        params['after'] = after

                    try:
                        # Send request with timeout
                        response = session.get(url, params=params, timeout=30)

                        # Check if request worked
                        if response.status_code != 200:
                            print(f"    HTTP {response.status_code}, moving to next...")
                            break

                        # Parse JSON response
                        data = response.json()
                        posts = data['data']['children']

                        if not posts:
                            print(f"    No more posts found")
                            break

                        # Process each post
                        new_reviews_this_page = 0
                        for post in posts:
                            if len(all_reviews) >= target:
                                break

                            post_data = post['data']
                            post_id = post_data.get('id')

                            # Skip if already seen
                            if post_id in seen_ids:
                                continue

                            seen_ids.add(post_id)

                            # Get title and content
                            title = post_data.get('title', '')
                            content = post_data.get('selftext', '')
                            full_text = title + ' ' + content

                            # Only keep product reviews
                            if is_product_review(full_text):
                                category = get_category(full_text)
                                post_date = datetime.fromtimestamp(post_data.get('created_utc', 0))

                                # Store the review
                                all_reviews.append({
                                    'id': post_id,
                                    'title': title,
                                    'content': content,
                                    'category': category,
                                    'score': post_data.get('score', 0),
                                    'comments': post_data.get('num_comments', 0),
                                    'date': post_date,
                                    'author': post_data.get('author', 'unknown'),
                                    'year': post_date.year,
                                    'subreddit': subreddit,
                                    'scraped_at': datetime.now()
                                })
                                new_reviews_this_page += 1

                                # Show progress periodically
                                if len(all_reviews) % 25 == 0 and len(all_reviews) > 0:
                                    print(f"\n    Found {len(all_reviews)} reviews total!")
                                    # Save checkpoint every 25 reviews
                                    save_checkpoint(all_reviews, checkpoint_file)

                        # Update progress bar
                        pages += 1
                        pbar.update(1)

                        if new_reviews_this_page > 0:
                            pbar.set_postfix({"reviews": len(all_reviews), "new": new_reviews_this_page})

                        # Get token for next page
                        after = data['data'].get('after')
                        consecutive_errors = 0  # Reset error counter on success

                        if not after:
                            print(f"    Reached last page")
                            break

                        # Wait to respect Reddit's rate limits (gradual backoff)
                        time.sleep(min(1 + pages * 0.1, 3))

                    except requests.exceptions.Timeout:
                        consecutive_errors += 1
                        print(f"    Timeout (attempt {consecutive_errors}/3), retrying...")
                        time.sleep(5)
                        continue

                    except requests.exceptions.ConnectionError:
                        consecutive_errors += 1
                        print(f"    Connection error (attempt {consecutive_errors}/3), retrying...")
                        time.sleep(10)
                        continue

                    except Exception as e:
                        consecutive_errors += 1
                        print(f"    Error: {str(e)[:100]} (attempt {consecutive_errors}/3)")
                        time.sleep(5)
                        continue

                pbar.close()

                # Small delay between sort methods
                time.sleep(2)

            # Small delay between subreddits
            time.sleep(3)

        # Save checkpoint after each full attempt
        save_checkpoint(all_reviews, checkpoint_file)

        # If we didn't reach target, wait before next attempt
        if len(all_reviews) < target and attempt < max_attempts:
            print(f"\nWaiting 60 seconds before next attempt...")
            time.sleep(60)

    return all_reviews


# ================= Main execution =================================

if __name__ == "__main__":
    # ensure the data folder exists
    os.makedirs('data/raw', exist_ok=True)

    # target number of reviews
    TARGET = 500
    OUTPUT_FILE = 'data/raw/product_reviews.csv'

    # start scraping
    start_time = datetime.now()
    print(f"\nSTARTING SCRAPING AT {start_time.strftime('%H:%M:%S')}")
    print("   (This may take 10-30 minutes depending on network)\n")

    # run the scraper
    all_reviews = scrape_reddit_reviews(target=TARGET, checkpoint_file=OUTPUT_FILE)

    # calculate time taken
    end_time = datetime.now()
    duration = end_time - start_time

    # final results
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE!")
    print("=" * 60)

    # convert to DataFrame
    df = pd.DataFrame(all_reviews)

    print(f"\nFINAL RESULTS:")
    print(f"   Total reviews collected: {len(df)}")
    print(f"   Target (500): {'ACHIEVED' if len(df) >= TARGET else 'NOT REACHED'}")
    print(f"   Time taken: {duration.total_seconds() / 60:.1f} minutes")

    # show category breakdown
    if len(df) > 0:
        print("\nReviews by category:")
        for cat, count in df['category'].value_counts().items():
            pct = (count / len(df)) * 100
            bar = "█" * int(pct / 2)
            print(f"   {cat:12s}: {count:3d} ({pct:5.1f}%) {bar}")

        print("\nReviews by subreddit:")
        for sub, count in df['subreddit'].value_counts().items():
            print(f"   r/{sub:15s}: {count:3d} reviews")

    # Save final database
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"\nDatabase saved to: {OUTPUT_FILE}")
    print(f"   Total records: {len(df)}")
    print(f"   Total columns: {len(df.columns)}")

    # Save as JSON backup
    json_file = 'data/raw/product_reviews.json'
    df.to_json(json_file, orient='records', indent=2, date_format='iso')
    print(f"   Backup saved to: {json_file}")

    print("\n" + "=" * 60)
    print("WEB SCRAPING COMPLETE")
    print("=" * 60)

    if len(df) < TARGET:
        print(f"\nWARNING: Only collected {len(df)}/{TARGET} reviews.")
        print("   Suggestions:")
        print("   1. Run the script again - it will resume from where it stopped")
        print("   2. Check your internet connection")
        #print("   3. Try using a VPN if Reddit is blocking your IP")
        print("   3. Add more subreddits to the list in the script")
    else:
        print("\nSUCCESS! Ready for next step.")