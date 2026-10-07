"""
Code 04: Data Analysis

- We are going to analyze the cleaned data to extract meaningful insights, using:
1- statistical analysis (mean, sum and count)
2- frequency analysis (counter algo for word frequencies)
3- comparative analysis (grouping by category)
4- KNN Classification (K-Nearest Neighbors) to predict product category

Output: saves analysis to: outputs/reports/
"""

import pandas as pd
from collections import Counter  # Algorithm for counting word frequencies
from datetime import datetime
import json
import os

# KNN imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#force the script to always run from the current folder
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

#create outputs folder if it doesn't exist
os.makedirs('outputs/reports', exist_ok=True)

print("=" * 60)
print("STEP 3: DATA ANALYSIS")
print("=" * 60)
print("")


# 1. load the cleaned data 
print("LOADING CLEANED DATABASE")
print("-" * 40)

df = pd.read_csv('data/cleaned/product_reviews_cleaned.csv', encoding='utf-8-sig')
df['date'] = pd.to_datetime(df['date'])

print(f"Loaded {len(df)} records")
print("")



# 2. statistical analysis 
print("=" * 50)
print("ANALYSIS 1: STATISTICAL ANALYSIS")
print("=" * 50)

total = len(df)
avg_score = df['score'].mean()
max_score = df['score'].max()
avg_comments = df['comments'].mean()
total_comments = df['comments'].sum()
avg_length = df['text_length'].mean()

print(f"\nBasic Statistics:")
print(f"  - Total reviews: {total}")
print(f"  - Average score (upvotes): {avg_score:.2f}")
print(f"  - Highest score: {max_score}")
print(f"  - Average comments: {avg_comments:.2f}")
print(f"  - Total comments: {total_comments}")
print(f"  - Average review length: {avg_length:.1f} characters")



# 3. category analysis
print(f"\nCategory Statistics:")
cat_counts = df['category'].value_counts()
most_discussed = cat_counts.index[0]

for cat, count in cat_counts.items():
    pct = (count / total) * 100
    cat_avg_score = df[df['category'] == cat]['score'].mean()
    print(f"  - {cat}: {count} ({pct:.1f}%), avg score: {cat_avg_score:.1f}")

print(f"\n  Most discussed category: {most_discussed}")


# 4. word frequency analysis
print("\n" + "=" * 50)
print("ANALYSIS 2: WORD FREQUENCY ANALYSIS")
print("=" * 50)
print("Algorithm used: Python's Counter")

# Combine all text
all_text = ' '.join(df['full_text'].dropna())

# Split into words
words = all_text.split()

# Count frequency using Counter algorithm
word_freq = Counter(words)

# Get top 15 words
top_words = word_freq.most_common(15)

print("\nTop 15 Most Frequent Words:")
print("-" * 40)
for i, (word, count) in enumerate(top_words, 1):
    print(f"  {i:2d}. '{word}': {count} times")




# 5. category specific analysis
print("\n" + "=" * 50)
print("ANALYSIS 3: CATEGORY-SPECIFIC ANALYSIS")
print("=" * 50)

category_results = []

for category in df['category'].unique():
    cat_df = df[df['category'] == category]
    
    cat_count = len(cat_df)
    cat_pct = (cat_count / total) * 100
    cat_avg_score = cat_df['score'].mean()
    cat_avg_comments = cat_df['comments'].mean()
    
    # Find top words for this category
    cat_text = ' '.join(cat_df['full_text'].dropna())
    cat_words = cat_text.split()
    cat_word_freq = Counter(cat_words)
    cat_top_words = cat_word_freq.most_common(5)
    
    category_results.append({
        'category': category,
        'count': cat_count,
        'percentage': cat_pct,
        'avg_score': cat_avg_score,
        'avg_comments': cat_avg_comments,
        'top_words': cat_top_words
    })
    
    print(f"\n{category.upper()}:")
    print(f"  Reviews: {cat_count} ({cat_pct:.1f}%)")
    print(f"  Average score: {cat_avg_score:.1f}")
    print(f"  Top 5 words: {[w for w, c in cat_top_words]}")


# ================================================================
# 6. KNN CLASSIFICATION
# ================================================================
print("\n" + "=" * 60)
print("ANALYSIS 4: KNN CLASSIFICATION (K-Nearest Neighbors)")
print("=" * 60)
print("""
Goal: Use KNN to predict the CATEGORY of a Reddit post
      based on its numerical features (score, comments, text_length).

How KNN works:
  - For each new data point, KNN looks at the K nearest neighbors
    in the training data (using Euclidean distance).
  - It then assigns the most common category among those neighbors.
  - K=5 means we look at the 5 closest posts to make the prediction.
""")

# --- Prepare features and target ---
features = ['score', 'comments', 'text_length']
target = 'category'

# Drop rows with missing values in these columns
knn_df = df[features + [target]].dropna()

X = knn_df[features]
y = knn_df[target]

# Encode labels (category names -> numbers)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalize features (important for KNN: distance-based algorithm)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

print(f"Dataset split:")
print(f"  - Training samples : {len(X_train)}")
print(f"  - Testing  samples : {len(X_test)}")

# --- Train KNN model ---
K = 5
print(f"\nTraining KNN model with K={K}...")
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train, y_train)
print("Model trained successfully.")

# --- Evaluate ---
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*40}")
print(f"  KNN RESULTS (K={K})")
print(f"{'='*40}")
print(f"  Overall Accuracy: {accuracy * 100:.2f}%")

print(f"\nDetailed Classification Report:")
print("-" * 40)
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_,
    zero_division=0
))
"""
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"  (Rows = Actual, Columns = Predicted)")
print(f"  Categories: {list(le.classes_)}")
print(cm)
"""

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"  (Rows = Actual, Columns = Predicted)")
print(f"  Categories: {list(le.classes_)}")
print(cm)

# ---- NEW: Save Confusion Matrix as PNG ----
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('outputs/charts', exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 6))

# Draw the heatmap manually
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)

# Labels
classes = le.classes_
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(classes, fontsize=11)

# Add numbers inside each cell
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha='center', va='center', fontsize=13, fontweight='bold',
                color='white' if cm[i, j] > thresh else 'black')

ax.set_title(f'Confusion Matrix (KNN, K={K})', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Actual Category', fontsize=12)
ax.set_xlabel('Predicted Category', fontsize=12)

plt.tight_layout()
plt.savefig('outputs/charts/07_confusion_matrix.png', dpi=150)
plt.close()
print("Saved: outputs/charts/07_confusion_matrix.png")


# ---- NEW: Save Classification Report as PNG ----
report_dict = classification_report(
    y_test, y_pred,
    target_names=le.classes_,
    zero_division=0,
    output_dict=True  # returns a dictionary instead of a string
)

# Convert to DataFrame for easy plotting
report_df = pd.DataFrame(report_dict).transpose()

# Keep only the per-category rows (not averages)
categories_only = report_df.loc[le.classes_, ['precision', 'recall', 'f1-score']]

fig, ax = plt.subplots(figsize=(9, 5))
ax.axis('off')  # hide the axes, we only want the table

# Build the table
col_labels = ['Category', 'Precision', 'Recall', 'F1-Score']
table_data = []
for cat in le.classes_:
    row = categories_only.loc[cat]
    table_data.append([
        cat,
        f"{row['precision']:.2f}",
        f"{row['recall']:.2f}",
        f"{row['f1-score']:.2f}"
    ])

# Add overall accuracy row at the bottom
table_data.append(['— Overall Accuracy —', '', '', f"{accuracy * 100:.2f}%"])

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.4, 2.2)  # (width, height) scaling

# Style the header row
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#2C7BB6')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Style category rows with alternating colors
for i in range(1, len(table_data)):
    for j in range(len(col_labels)):
        if i == len(table_data):  # last row (accuracy)
            table[i, j].set_facecolor('#F0F0F0')
            table[i, j].set_text_props(fontweight='bold')
        elif i % 2 == 0:
            table[i, j].set_facecolor('#EAF4FB')
        else:
            table[i, j].set_facecolor('white')

ax.set_title(f'Classification Report (KNN, K={K})', fontsize=14,
             fontweight='bold', pad=20, y=0.98)

plt.tight_layout()
plt.savefig('outputs/charts/08_classification_report.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/charts/08_classification_report.png")

# --- Try different K values to find best K ---
print(f"\n{'='*40}")
print("Finding best K value (K=1 to 15):")
print("-" * 40)

best_k = K
best_acc = accuracy

for k in range(1, 16):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    marker = " <-- best" if k == 1 else ""
    if acc > best_acc:
        best_acc = acc
        best_k = k
        marker = " <-- best"
    elif acc == best_acc and k == best_k:
        marker = " <-- best"
    else:
        marker = ""
    print(f"  K={k:2d} -> Accuracy: {acc * 100:.2f}%{marker}")

print(f"\n  Best K found: K={best_k} with accuracy {best_acc * 100:.2f}%")

# --- Example prediction ---
print(f"\n{'='*40}")
print("Example Prediction with KNN:")
print("-" * 40)
example = pd.DataFrame([[10, 3, 250]], columns=features)
example_scaled = scaler.transform(example)
pred_encoded = knn.predict(example_scaled)
pred_category = le.inverse_transform(pred_encoded)[0]
print(f"  New post  -> score=10, comments=3, text_length=250")
print(f"  Predicted category: '{pred_category}'")

# Save KNN results
knn_results = []
for k in range(1, 16):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    knn_results.append({'K': k, 'accuracy': round(acc * 100, 2)})

knn_df_results = pd.DataFrame(knn_results)
knn_df_results.to_csv('outputs/reports/knn_accuracy_by_k.csv', index=False, encoding='utf-8-sig')
print(f"\nSaved: outputs/reports/knn_accuracy_by_k.csv")


# 7. save analysis to results
print("\n" + "=" * 50)
print("SAVING ANALYSIS RESULTS")
print("=" * 50)

# Save summary statistics
summary_df = pd.DataFrame([
    ['total_reviews', total, 'integer'],
    ['average_score', avg_score, 'float'],
    ['highest_score', max_score, 'integer'],
    ['average_comments', avg_comments, 'float'],
    ['total_comments', total_comments, 'integer'],
    ['average_length', avg_length, 'float'],
    ['most_discussed_category', most_discussed, 'string'],
    ['knn_accuracy_k5', round(accuracy * 100, 2), 'float'],
    ['knn_best_k', best_k, 'integer'],
    ['knn_best_accuracy', round(best_acc * 100, 2), 'float'],
], columns=['metric', 'value', 'type'])

summary_df.to_csv('outputs/reports/analysis_summary.csv', index=False, encoding='utf-8-sig')
print("Saved: outputs/reports/analysis_summary.csv")

# Save top words
top_words_df = pd.DataFrame(top_words, columns=['word', 'frequency'])
top_words_df.to_csv('outputs/reports/top_words.csv', index=False, encoding='utf-8-sig')
print("Saved: outputs/reports/top_words.csv")

# Save category analysis
cat_df_out = pd.DataFrame(category_results)
cat_df_out.to_csv('outputs/reports/category_analysis.csv', index=False, encoding='utf-8-sig')
print("Saved: outputs/reports/category_analysis.csv")



#============ GENERATE REPORT ===================
print("\nGENERATING TEXT REPORT")
print("-" * 40)

report = f"""
========================================
   ANALYSIS REPORT
   Generated: {datetime.now()}
========================================

1. DATA COLLECTION
   - Total reviews: {total}
   - Source: r/algeriarates and r/algeria

2. BASIC STATISTICS
   - Average score: {avg_score:.2f}
   - Highest score: {max_score}
   - Average comments: {avg_comments:.2f}
   - Most discussed category: {most_discussed}

3. CATEGORY DISTRIBUTION
"""
for cat, count in cat_counts.items():
    pct = (count / total) * 100
    report += f"   - {cat}: {count} ({pct:.1f}%)\n"

report += """
4. MOST FREQUENT WORDS (TOP 15)
"""
for i, (word, count) in enumerate(top_words, 1):
    report += f"   {i}. '{word}': {count} times\n"

report += """
5. CATEGORY ANALYSIS
"""
for cat in category_results:
    report += f"""
   {cat['category'].upper()}:
   - Reviews: {cat['count']} ({cat['percentage']:.1f}%)
   - Average score: {cat['avg_score']:.2f}
   - Top words: {', '.join([w for w, c in cat['top_words']])}
"""

report += f"""
6. KNN CLASSIFICATION (K-Nearest Neighbors)
   - Algorithm: KNN (supervised classification)
   - Features used: score, comments, text_length
   - Target: category (product type)
   - Train/Test split: 80% / 20%
   - K value used: {K}
   - Accuracy with K={K}: {accuracy * 100:.2f}%
   - Best K found: K={best_k} (accuracy: {best_acc * 100:.2f}%)

========================================
   END OF REPORT
========================================
"""

with open('outputs/reports/analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("Saved: outputs/reports/analysis_report.txt")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\nNEXT")
