"""Code 02: Data Cleaning and Preprocessing

- We are going to clean the collected data using these techniques:
1- Handling missing values (removing empty rows, filling missing data)
2- Removing duplicates (posts with the same titles and content)
3- Removing stopwords 
4- Correcting format errors (lowercase, removing URLs, special chars)

Output: Saved cleaned data to: data/cleaned/product_reviews_cleaned.csv
"""

import os
import pandas as pd  # For data manipulation
import re  # For text cleaning (regular expressions)

#force the script to always run from the current folder
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

#create output folder if it doesn't exist
os.makedirs('data/cleaned', exist_ok=True)


# 1. define the stop words
STOP_WORDS = [
    # English stop words
    'the', 'a', 'an', 'and', 'or', 'but', 'so', 'for', 'of', 'to', 'in',
    'on', 'at', 'by', 'with', 'without', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'will', 'would', 'could', 'should', 'this', 'that',
    'these', 'those', 'it', 'they', 'we', 'you', 'he', 'she', 'them',
    'from', 'up', 'down', 'out', 'off', 'over', 'under', 'then', 'once',
    'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'than', 'too', 'very', 'just', 'as', 's', 'm', 't',

    # French stop words (common in Algerian posts)
    'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou',
    'mais', 'donc', 'car', 'ni', 'pour', 'par', 'avec', 'sans',
    'sous', 'sur', 'dans', 'hors', 'est', 'sont', 'était', 'étaient',
    'ce', 'cette', 'ces', 'il', 'elle', 'ils', 'elles', 'nous', 'vous',

    # Common review words (not meaningful)
    'bon', 'bien', 'très', 'trop', 'peu', 'beaucoup', 'juste', 'fait',
    'faire', 'peut', 'être', 'avoir', 'good', 'bad', 'nice', 'great',
    'well', 'just', 'like', 'product', 'buy', 'price', 'quality'
]


print("=" * 60)
print("STEP 2: DATA CLEANING AND PREPROCESSING")
print("=" * 60)
print(f"Loaded {len(STOP_WORDS)} stop words")
print("")


# 2. define cleaning functions

def clean_text(text):
    """
    Clean text by removing unwanted characters.
    Steps: lowercase, remove URLs, remove special char and lastly remove extra spaces
    """
    if pd.isna(text):  # Handle missing values
        return ""

    text = str(text).lower()  # Convert to lowercase

    # Remove URLs (http://, https://, www.)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove mentions (@username) and hashtags (#topic)
    text = re.sub(r'@\w+|#\w+', '', text)

    # Keep only letters, numbers, and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stop_words(text):
    """
    Remove stop words 
    """
    words = text.split()
    filtered = [w for w in words if w not in STOP_WORDS]
    return ' '.join(filtered)


def handle_missing_values(df):
    """Handle missing values in the database."""
    print("\n1. HANDLING MISSING VALUES")
    print("-" * 40)

    before = len(df)

    # remove rows with empty content
    df = df[df['content'].notna()]
    df = df[df['content'].str.len() > 10]
    print(f"   Removed {before - len(df)} rows with empty content")

    # fill missing values with defaults
    df['score'] = df['score'].fillna(0)
    df['comments'] = df['comments'].fillna(0)
    df['author'] = df['author'].fillna('unknown')

    print(f"   Filled missing values (score=0, comments=0, author='unknown')")
    return df


def remove_duplicates(df):
    """Remove duplicate posts (same title and content)."""
    print("\n2. REMOVING DUPLICATES")
    print("-" * 40)

    before = len(df)
    df = df.drop_duplicates(subset=['title', 'content'])
    print(f"   Removed {before - len(df)} duplicate posts")
    return df


# 3. load the raw data

print("LOADING RAW DATABASE")
print("-" * 40)

df = pd.read_csv('data/raw/product_reviews.csv', encoding='utf-8-sig')
df['date'] = pd.to_datetime(df['date'])

#save original count before cleaning (used in summary at the end)
original_count = len(df)

print(f"Loaded {len(df)} records")
print("")


# 4. apply the cleaning techniques (mentioned in the start of the page)

# Technique 1: Handle missing values
df = handle_missing_values(df)

# Technique 2: Remove duplicates
df = remove_duplicates(df)

# Technique 3: Clean text (format correction)
print("\n3. CORRECTING FORMAT ERRORS")
print("-" * 40)
print("   Cleaning titles...")
df['clean_title'] = df['title'].apply(clean_text)
print("   Cleaning content...")
df['clean_content'] = df['content'].apply(clean_text)

# Technique 4: Remove stop words
print("\n4. REMOVING STOP WORDS")
print("-" * 40)
print("   Removing stop words from titles...")
df['clean_title'] = df['clean_title'].apply(remove_stop_words)
print("   Removing stop words from content...")
df['clean_content'] = df['clean_content'].apply(remove_stop_words)

# Add useful columns for analysis
df['full_text'] = df['clean_title'] + ' ' + df['clean_content']
df['text_length'] = df['clean_content'].str.len()


# 5. save the cleaned data

print("\n" + "=" * 40)
print("CLEANING SUMMARY")
print("=" * 40)
print(f"Original records: {original_count}")
print(f"Cleaned records: {len(df)}")
print(f"Records removed: {original_count - len(df)}")

print("\nCATEGORY DISTRIBUTION AFTER CLEANING:")
for cat, count in df['category'].value_counts().items():
    pct = (count / len(df)) * 100
    print(f"  {cat}: {count} ({pct:.1f}%)")

# save to csv
output_file = 'data/cleaned/product_reviews_cleaned.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\nCleaned database saved to: {output_file}")

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETE")
print("=" * 60)
print("\n NEXT")