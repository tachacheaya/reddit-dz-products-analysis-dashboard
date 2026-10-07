"""
Code 03: Showing Preprocessing (comparaison)
- We are going to show the difference between the data BEFORE and AFTER the preprocessing(cleaning)
- It shows:
1- the database size (nbr of rows and columns)
2- sample of text before vs after cleaning
3- category distribution before vs after
4- missing values handles
5- duplicated removed

Output: displays the comparaison in the console

"""

import pandas as pd
import os

#force the script to always run from the current folder
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("=" * 60)
print("PREPROCESSING COMPARISON - BEFORE VS AFTER")
print("=" * 60)
print("")


# 1. raw data BEFORE cleaning
raw_df = pd.read_csv('data/raw/product_reviews.csv', encoding='utf-8-sig')

#cleaned data AFTER cleaning 
cleaned_df = pd.read_csv('data/cleaned/product_reviews_cleaned.csv', encoding='utf-8-sig')

print(f"Raw data:     {len(raw_df)} rows, {len(raw_df.columns)} columns")
print(f"Cleaned data: {len(cleaned_df)} rows, {len(cleaned_df.columns)} columns")
print(f"Removed:      {len(raw_df) - len(cleaned_df)} rows")
print("")


# 2. BDD size 
print("=" * 50)
print("1. DATABASE SIZE (Taille de la base de données)")
print("=" * 50)

print(f"\nBEFORE CLEANING:")
print(f"  - Number of rows (records): {len(raw_df)}")
print(f"  - Number of columns (fields): {len(raw_df.columns)}")
print(f"  - Columns: {list(raw_df.columns)}")

print(f"\nAFTER CLEANING:")
print(f"  - Number of rows (records): {len(cleaned_df)}")
print(f"  - Number of columns (fields): {len(cleaned_df.columns)}")
print(f"  - Columns: {list(cleaned_df.columns)}")

print(f"\nNEW COLUMNS ADDED:")
new_cols = set(cleaned_df.columns) - set(raw_df.columns)
for col in sorted(new_cols):
    print(f"  + {col}")



# 3. text transformation sample
print("\n" + "=" * 50)
print("2. TEXT TRANSFORMATION EXAMPLE")
print("=" * 50)

# find a review with content
sample_idx = 0
for i in range(len(raw_df)):
    if len(str(raw_df.iloc[i]['content'])) > 50:
        sample_idx = i
        break

print("\nBEFORE CLEANING:")
print("-" * 40)
print(f"Title: {raw_df.iloc[sample_idx]['title'][:150]}...")
print(f"Content: {str(raw_df.iloc[sample_idx]['content'])[:200]}...")

print("\nAFTER CLEANING:")
print("-" * 40)
print(f"Title: {cleaned_df.iloc[sample_idx]['clean_title'][:150]}...")
print(f"Content: {cleaned_df.iloc[sample_idx]['clean_content'][:200]}...")

print("\nWHAT CHANGED:")
print("  - Text converted to lowercase")
print("  - URLs and special characters removed")
print("  - Stop words removed (the, a, and, etc.)")
print("  - Extra spaces removed")



# 4. category distribution before vs after 
print("\n" + "=" * 50)
print("3. CATEGORY DISTRIBUTION (Avant vs Après)")
print("=" * 50)

raw_cats = raw_df['category'].value_counts()
cleaned_cats = cleaned_df['category'].value_counts()

print("\nCategory      | BEFORE | AFTER | Change")
print("-" * 45)
for cat in sorted(set(raw_cats.index) | set(cleaned_cats.index)):
    before = raw_cats.get(cat, 0)
    after = cleaned_cats.get(cat, 0)
    change = after - before
    print(f"{cat:12} | {before:6} | {after:5} | {change:+d}")



# 5. missing values 
print("\n" + "=" * 50)
print("4. MISSING VALUES HANDLED")
print("=" * 50)

raw_missing = raw_df.isnull().sum()
print("\nBEFORE CLEANING:")
if raw_missing.sum() > 0:
    for col in raw_missing[raw_missing > 0].index:
        print(f"  {col}: {raw_missing[col]} missing values")
else:
    print("  No missing values found")

cleaned_missing = cleaned_df.isnull().sum()
print("\nAFTER CLEANING:")
if cleaned_missing.sum() > 0:
    for col in cleaned_missing[cleaned_missing > 0].index:
        print(f"  {col}: {cleaned_missing[col]} missing values")
else:
    print("  No missing values remaining")



# 6. duplicates removed
print("\n" + "=" * 50)
print("5. DUPLICATES REMOVED")
print("=" * 50)

duplicates = raw_df.duplicated(subset=['title', 'content']).sum()
print(f"Duplicates found in raw data: {duplicates}")
print(f"All duplicates were removed during cleaning")


#=======================================
print("\n" + "=" * 60)
print("SUMMARY OF PREPROCESSING")
print("=" * 60)

print("""
The following preprocessing techniques were applied:

1. MISSING VALUES HANDLING:
   - Removed rows with empty content
   - Filled missing scores with 0
   - Filled missing authors with 'unknown'

2. DUPLICATE REMOVAL:
   - Removed posts with identical title and content

3. TEXT CLEANING (Format Correction):
   - Converted to lowercase
   - Removed URLs, special characters, mentions, hashtags
   - Removed extra spaces

4. STOP WORDS REMOVAL:
   - Removed common English and French stop words
   - Removed review-specific common words
""")

print("\n" + "=" * 60)
print("PREPROCESSING COMPARISON COMPLETE")
print("=" * 60)
print("\n NEXT")