"""
Code 05: Data Visualization

- We are going to create charts to visualize the analysis results.
They are: 
1. Category distribution (bar chart) - shows how many reviews per category
2. Category distribution (pie chart) - shows percentage per category
3. Average scores by category (bar chart) - shows which categories get highest upvotes
4. Average comments by category (bar chart) - shows which categories generate discussion
5. Average review length by category (bar chart) - shows which categories have more detailed reviews
6. Top products/brands mentioned (horizontal bar chart) - shows most mentioned products

Output: saves the chart to: outputs/charts/

"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

#force the script to always run from the current folder
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Create outputs/charts folder if it doesn't exist
os.makedirs('outputs/charts', exist_ok=True)

# Set style for charts
plt.style.use('default')

print("=" * 60)
print("STEP 5: DATA VISUALIZATION")
print("=" * 60)
print("")

# 1. load the cleaned data
print("LOADING DATA")
print("-" * 40)

df = pd.read_csv('data/cleaned/product_reviews_cleaned.csv', encoding='utf-8-sig')

print(f"Loaded {len(df)} records")
print("")


# Calculate review length (number of characters in content + title)
df['review_length'] = df['title'].fillna('').astype(str).str.len() + df['content'].fillna('').astype(str).str.len()

# ============================================================================
# CHART 1: Category Distribution (Bar Chart)
# ============================================================================

print("CREATING CHARTS")
print("-" * 40)
print("1. Category distribution bar chart...")

cat_counts = df['category'].value_counts()

plt.figure(figsize=(10, 6))
bars = plt.bar(cat_counts.index, cat_counts.values, color='steelblue', edgecolor='black')

# Add numbers on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.title('Number of Product Reviews by Category', fontsize=14, fontweight='bold')
plt.xlabel('Product Category')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/charts/01_category_distribution.png', dpi=150)
plt.close()
print("   Saved: 01_category_distribution.png")

# ============================================================================
# CHART 2: Category Distribution (Pie Chart)
# ============================================================================

print("2. Category distribution pie chart...")

plt.figure(figsize=(8, 8))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
plt.pie(cat_counts.values, labels=cat_counts.index, autopct='%1.1f%%',
        colors=colors[:len(cat_counts)], startangle=90)
plt.title('Product Reviews Distribution by Category', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/charts/02_category_pie.png', dpi=150)
plt.close()
print("   Saved: 02_category_pie.png")

# ============================================================================
# CHART 3: Average Scores by Category
# ============================================================================

print("3. Average scores by category...")

avg_scores = df.groupby('category')['score'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(avg_scores.index, avg_scores.values, color='coral', edgecolor='black')

# Add numbers on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

plt.title('Average Upvote Score by Category', fontsize=14, fontweight='bold')
plt.xlabel('Product Category')
plt.ylabel('Average Score (Upvotes)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/charts/03_avg_scores_by_category.png', dpi=150)
plt.close()
print("   Saved: 03_avg_scores_by_category.png")

# ============================================================================
# CHART 4: Average Comments by Category
# ============================================================================

print("4. Average comments by category...")

avg_comments = df.groupby('category')['comments'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(avg_comments.index, avg_comments.values, color='seagreen', edgecolor='black')

# Add numbers on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

plt.title('Average Number of Comments by Category', fontsize=14, fontweight='bold')
plt.xlabel('Product Category')
plt.ylabel('Average Comments')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/charts/04_avg_comments_by_category.png', dpi=150)
plt.close()
print("   Saved: 04_avg_comments_by_category.png")

# ============================================================================
# CHART 5: Average Review Length by Category
# ============================================================================

print("5. Average review length by category...")

avg_length = df.groupby('category')['review_length'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(avg_length.index, avg_length.values, color='mediumpurple', edgecolor='black')

# Add numbers on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.0f}', ha='center', va='bottom', fontweight='bold')

plt.title('Average Review Length by Category (characters)', fontsize=14, fontweight='bold')
plt.xlabel('Product Category')
plt.ylabel('Average Review Length (characters)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/charts/05_avg_review_length_by_category.png', dpi=150)
plt.close()
print("   Saved: 05_avg_review_length_by_category.png")

# ============================================================================
# CHART 6: Top Products/Brands Mentioned
# ============================================================================

print("6. Top products/brands mentioned...")

# Define product keywords to search for in reviews
product_keywords = {
    'Condor': ['condor'],
    'Samsung': ['samsung'],
    'Xiaomi': ['xiaomi'],
    'LG': ['lg'],
    'Rouiba': ['rouiba'],
    'Mobilis': ['mobilis'],
    'Ooredoo': ['ooredoo'],
    'Djezzy': ['djezzy'],
    'Nivea': ['nivea'],
    'Loreal': ['loreal'],
    'Garnier': ['garnier'],
    'Candia': ['candia'],
    'Tchin-lait': ['tchin-lait'],
    'Hamoud': ['hamoud'],
    'Ifri': ['ifri']
}

# Count mentions of each product/brand
mentions = {}
for product, keywords in product_keywords.items():
    count = 0
    for keyword in keywords:
        count += df['content'].fillna('').astype(str).str.lower().str.contains(keyword).sum()
        count += df['title'].fillna('').astype(str).str.lower().str.contains(keyword).sum()
    if count > 0:
        mentions[product] = count

# Sort and get top 10
sorted_mentions = dict(sorted(mentions.items(), key=lambda x: x[1], reverse=True)[:10])

if sorted_mentions:
    plt.figure(figsize=(12, 8))
    bars = plt.barh(list(sorted_mentions.keys()), list(sorted_mentions.values()), 
                    color='lightcoral', edgecolor='black')
    
    # Add numbers on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2., 
                 f'{int(width)}', ha='left', va='center', fontweight='bold')
    
    plt.title('Top 10 Most Mentioned Products/Brands', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Mentions')
    plt.ylabel('Product/Brand')
    plt.gca().invert_yaxis()  # Highest at top
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/charts/06_top_products_brands.png', dpi=150)
    plt.close()
    print("   Saved: 06_top_products_brands.png")
else:
    print("   No product mentions found to display")

# ============================================================================
# CHART 7: Engagement Ratio by Category (Comments per Score)
# ============================================================================
"""
print("7. Engagement ratio by category...")

# Calculate engagement ratio (comments per upvote, avoid division by zero)
df['engagement_ratio'] = df['comments'] / (df['score'] + 1)  # +1 to avoid division by zero
engagement_by_category = df.groupby('category')['engagement_ratio'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(engagement_by_category.index, engagement_by_category.values, 
               color='teal', edgecolor='black')

# Add numbers on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

plt.title('Engagement Ratio by Category (Comments per Upvote)', fontsize=14, fontweight='bold')
plt.xlabel('Product Category')
plt.ylabel('Comments per Upvote Ratio')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add explanation text
plt.figtext(0.5, 0.01, 'Lower ratio = more upvotes than comments (popular but less discussion)\nHigher ratio = more discussion relative to upvotes', 
            ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('outputs/charts/07_engagement_ratio_by_category.png', dpi=150)
plt.close()
print("   Saved: 07_engagement_ratio_by_category.png")

"""
# ============================================================================
# CHART 7: Top Brands per Category (4 subplots in one PNG)
# ============================================================================

print(".9 Top brands per category (4 subplots)...")

# Define which brands belong to which category
category_brands = {
    'food': {
        'Rouiba':     ['rouiba'],
        'Hamoud':     ['hamoud'],
        'Candia':     ['candia'],
        'Ifri':       ['ifri'],
        'Tchin-lait': ['tchin-lait', 'tchin lait'],
        'Bifa': ['bifa'],
        'Bimo': ['bimo'],
        'Tazej': ['tazej'],
        'Kool': ['kool'],
    },
    'electronics': {
        'Condor':  ['condor'],
        'Samsung': ['samsung'],
        'Xiaomi':  ['xiaomi'],
        'Mobilis': ['mobilis'],
        'Djezzy':  ['djezzy'],
        'Ooredoo': ['ooredoo'],
    },
    'appliances': {
        'Condor':    ['condor'],
        'LG':        ['lg'],
        'Samsung':   ['samsung'],
        'Hisense':   ['hisense'],
        'Whirlpool': ['whirlpool'],
        'Sony':      ['sony'],
    },
    'cosmetics': {
        'Nivea':   ['nivea'],
        'Garnier': ['garnier'],
        'Loreal':  ['loreal'],
        'Dove':    ['dove'],
        'Dettol':  ['dettol'],
        'Touché': ["touché"],
        'Natribifluor':    ['natribifluor'],
        'Felari':          ['felari'],
        'Burberry':        ['burberry'],
        'Emmanuelle Jane': ['emmanuelle jane'],
        'COSRX':           ['cosrx'],
    }
}

# Colors for each category subplot
subplot_colors = {
    'food':        '#FF6B6B',
    'electronics': '#45B7D1',
    'appliances':  '#96CEB4',
    'cosmetics':   '#FFA07A',
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Most Mentioned Brands by Product Category',
             fontsize=16, fontweight='bold', y=1.01)

# Flatten axes for easy looping: [[ax1,ax2],[ax3,ax4]] -> [ax1,ax2,ax3,ax4]
axes_flat = axes.flatten()

for idx, (category, brands) in enumerate(category_brands.items()):
    ax = axes_flat[idx]

    # Filter df to this category only
    cat_df = df[df['category'] == category]

    # Count mentions of each brand in this category's posts
    brand_counts = {}
    for brand, keywords in brands.items():
        count = 0
        for keyword in keywords:
            count += cat_df['content'].fillna('').astype(str).str.lower().str.contains(keyword).sum()
            count += cat_df['title'].fillna('').astype(str).str.lower().str.contains(keyword).sum()
        if count > 0:
            brand_counts[brand] = count

    if not brand_counts:
        ax.text(0.5, 0.5, 'No brand mentions found',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(category.upper(), fontsize=13, fontweight='bold')
        continue

    # Sort brands by count
    sorted_brands = dict(sorted(brand_counts.items(), key=lambda x: x[1], reverse=True))

    bars = ax.barh(list(sorted_brands.keys()),
                   list(sorted_brands.values()),
                   color=subplot_colors[category],
                   edgecolor='black')

    # Add count labels on each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{int(width)}', va='center', fontsize=10, fontweight='bold')

    ax.set_title(category.upper(), fontsize=13, fontweight='bold',
                 color=subplot_colors[category])
    ax.set_xlabel('Number of Mentions', fontsize=10)
    ax.invert_yaxis()  # highest at top
    ax.grid(axis='x', alpha=0.3)

    # Set x limit a bit wider so labels don't get cut off
    ax.set_xlim(0, max(sorted_brands.values()) * 1.2)

plt.tight_layout()
plt.savefig('outputs/charts/07_brands_per_category.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 07_brands_per_category.png")
# ===================== Summary =========================

print("")
print("=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)
print(f"\nTotal charts created: 7")
print(f"All charts saved to: outputs/charts/")
print("\nCharts created:")
print("  1. 01_category_distribution.png - Bar chart of review counts")
print("  2. 02_category_pie.png - Pie chart of category percentages")
print("  3. 03_avg_scores_by_category.png - Average upvotes per category")
print("  4. 04_avg_comments_by_category.png - Average comments per category")
print("  5. 05_avg_review_length_by_category.png - How detailed reviews are per category")
print("  6. 06_top_products_brands.png - Most mentioned products/brands")
#print("  7. 07_engagement_ratio_by_category.png - Discussion vs popularity ratio")

print("")
print("=" * 60)