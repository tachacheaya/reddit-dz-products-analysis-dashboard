import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from collections import Counter
import re
from pathlib import Path

# ─── Load & Prepare Data ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "data" / "cleaned" / "product_reviews_cleaned.csv")
df['date']       = pd.to_datetime(df['date'])
df['month_name'] = df['date'].dt.strftime('%b')
df['month_num']  = df['date'].dt.month
df['day_name']   = df['date'].dt.day_name()
df['engagement'] = df['score'] + df['comments']
knn_df = pd.read_csv(BASE_DIR / "outputs" / "reports" / "knn_accuracy_by_k.csv")

STOPWORDS = {
    'i','my','me','if','what','can','how','about','the','is','and','to','a',
    'in','it','for','of','that','this','be','have','has','was','are','we',
    'you','your','or','not','on','do','its','an','but','at','as','with',
    'they','their','any','all','also','so','very','just','more','by','our',
    'no','from','there','up','out','were','would','which','some','get','got','im',
    'don','doesn','really','know','like','use','one','want','good','best','new',
    'will','need','please','looking','anyone','think','something','make','used',
    'find','recommend','help','try','buy','price','product','algeria','algerian',
}
CATEGORIES = sorted(df['category'].unique())

# ─── Color System — Light Pink Editorial ────────────────────────────────────
C = {
    'bg':          '#FDE8EF',
    'bg2':         '#FAD4E0',
    'card':        'rgba(255,255,255,0.72)',
    'card_solid':  '#FFFFFF',
    'panel':       'rgba(255,255,255,0.55)',
    'border':      'rgba(220,100,140,0.18)',
    'border_med':  'rgba(220,100,140,0.30)',
    'ink':         '#2A1120',
    'ink_mid':     '#7A3A55',
    'ink_soft':    '#B07090',
    'grid':        'rgba(220,100,140,0.10)',
    'magenta':     '#E8186D',
    'coral':       '#FF5C5C',
    'tangerine':   '#FF8C42',
    'gold':        '#FFCA3A',
    'mint':        '#2DC653',
    'teal':        '#14B8A6',
    'violet':      '#7C3AED',
    'orchid':      '#D946EF',
    'sky':         '#0EA5E9',
    'rose':        '#FB7185',
}

CAT_COLORS = {
    'appliances':  '#E8186D',
    'food':        '#FF8C42',
    'electronics': '#14B8A6',
    'cosmetics':   '#D946EF',
}

# ─── Chart metadata for the modal ───────────────────────────────────────────
# Each entry: title, description, attributes used (list), visual channels (list)
CHART_META = {
    'fig-donut': {
        'title': 'Category Distribution',
        'icon': '🍩',
        'accent': '#E8186D',
        'description': (
            'This donut chart shows how the 430 Reddit posts are split across the four product categories: '
            'Appliances, Food, Electronics, and Cosmetics. '
            'The large central number shows the total post count for the current filter. '
            'Each slice represents one category\'s share of the dataset, making it easy to see at a glance '
            'that Appliances dominate Algerian consumer discourse on Reddit.'
        ),
        'attributes': [
            ('category', 'Categorical — Nominal', '#E8186D'),
            ('post count (derived)', 'Quantitative — Discrete', '#14B8A6'),
        ],
        'channels': [
            ('🎨 Color (hue)', 'One fixed hue per category — since categories have no inherent order, hue is the right choice.'),
            ('⭕ Arc angle / area', 'Slice size encodes the count proportion. Larger slice = more posts in that category.'),
            ('📝 Text label', 'Category name + percentage printed on each slice for direct reading without a legend lookup.'),
            ('🔢 Central annotation', 'Total post count anchored in the hole — gives a quick numeric summary without a separate KPI card.'),
        ],
    },
    'fig-score-bar': {
        'title': 'Average Score by Category',
        'icon': '📊',
        'accent': '#FF8C42',
        'description': (
            'A horizontal bar chart ranking the four product categories by their average Reddit score (upvotes). '
            'Food posts earn the highest average score (23.0), indicating that quality and availability topics '
            'resonate widely with the community. '
            'The horizontal orientation was chosen because category names are longer labels — they read more naturally on the Y axis.'
        ),
        'attributes': [
            ('category', 'Categorical — Nominal', '#E8186D'),
            ('score (averaged)', 'Quantitative — Continuous', '#7C3AED'),
        ],
        'channels': [
            ('📏 Bar length (X axis)', 'Length directly encodes the average score value — one of the most accurate visual channels for numeric comparison.'),
            ('🎨 Color (hue)', 'Each bar uses its category color for consistency with all other charts on the dashboard.'),
            ('📝 Value label', 'The numeric average is printed at the bar tip so users don\'t need to read the axis precisely.'),
            ('📍 Y position', 'Categories are sorted ascending by score — position encodes rank, letting the eye immediately spot the leader.'),
        ],
    },
    'fig-top-posts': {
        'title': 'Top 8 Posts by Score',
        'icon': '🏆',
        'accent': '#FF5C5C',
        'description': (
            'A ranked horizontal bar chart showing the eight highest-scoring individual Reddit posts. '
            'Each bar represents a single post, colored by its product category. '
            'This chart surfaces the specific conversations that resonated most with the community, '
            'complementing the category-level averages shown in the Score by Category chart.'
        ),
        'attributes': [
            ('title', 'Categorical — Nominal', '#E8186D'),
            ('score', 'Quantitative — Discrete', '#14B8A6'),
            ('category', 'Categorical — Nominal', '#D946EF'),
            ('comments', 'Quantitative — Discrete', '#FF8C42'),
        ],
        'channels': [
            ('📏 Bar length (X axis)', 'Score (upvote count) encoded as bar length — enables direct comparison between the top posts.'),
            ('🎨 Color (hue)', 'Category color applied to each bar — immediately shows which product type generates viral posts.'),
            ('📍 Y position (rank order)', 'Bars sorted descending so the top post is always at position #1 — rank is encoded by vertical position.'),
            ('📝 Hover tooltip', 'Full title, category, score, and comment count revealed on hover to avoid cluttering the chart with text.'),
        ],
    },
    'fig-timeline': {
        'title': 'Monthly Post Activity',
        'icon': '📈',
        'accent': '#14B8A6',
        'description': (
            'A multi-line spline chart tracking how many posts were published each month, '
            'broken down by product category. '
            'Time flows left-to-right on the X axis, and the smooth (spline) interpolation '
            'helps highlight trends rather than noise between individual months. '
            'Markers on each data point preserve the exact monthly values.'
        ),
        'attributes': [
            ('date (→ month)', 'Temporal', '#0EA5E9'),
            ('post count (derived)', 'Quantitative — Discrete', '#14B8A6'),
            ('category', 'Categorical — Nominal', '#E8186D'),
        ],
        'channels': [
            ('📍 X position (time)', 'Months ordered left-to-right on the X axis — temporal order is encoded by spatial position.'),
            ('📍 Y position (count)', 'Post count mapped to vertical position — higher = more posts that month.'),
            ('📈 Line slope', 'The slope between two months immediately signals whether activity is rising or falling.'),
            ('🎨 Color (hue)', 'Each category gets its fixed color — multiple lines on one chart without ambiguity.'),
            ('⭕ Markers', 'Small circles mark discrete monthly values, separating exact data points from the interpolated curve.'),
        ],
    },
    'fig-scatter': {
        'title': 'Score vs Comments',
        'icon': '🫧',
        'accent': '#7C3AED',
        'description': (
            'A bubble scatter plot comparing two engagement metrics — Reddit score (upvotes) and comment count — '
            'for every individual post. '
            'A third dimension, post text length, is encoded as bubble size: larger bubbles = longer posts. '
            'The X axis uses a logarithmic scale because scores span several orders of magnitude, '
            'which would crush low-scoring posts against the left edge on a linear scale.'
        ),
        'attributes': [
            ('score', 'Quantitative — Discrete', '#14B8A6'),
            ('comments', 'Quantitative — Discrete', '#FF8C42'),
            ('text_length', 'Quantitative — Continuous', '#7C3AED'),
            ('category', 'Categorical — Nominal', '#E8186D'),
            ('title', 'Categorical — Nominal', '#B07090'),
        ],
        'channels': [
            ('📍 X position (log scale)', 'Score encoded on log X — compresses the wide range so outliers don\'t dominate.'),
            ('📍 Y position', 'Comment count on linear Y axis — most posts cluster below 50 comments.'),
            ('⭕ Bubble size (area)', 'Text length → bubble area. Larger = more detailed post. Area is appropriate for continuous positive values.'),
            ('🎨 Color (hue)', 'Category color distinguishes the four groups without needing separate panels.'),
            ('📝 Hover tooltip', 'Post title, score, and comment count shown on hover to identify specific outliers.'),
        ],
    },
    'fig-keywords': {
        'title': 'Top Keywords in Posts',
        'icon': '🔤',
        'accent': '#D946EF',
        'description': (
            'A horizontal bar chart of the 12 most frequent meaningful words found across all post body texts, '
            'after removing common stopwords (English, French, and domain-specific filler). '
            'This gives a direct window into what topics and products Algerian consumers actually discuss. '
            'Each bar gets a distinct color from the dashboard palette to keep the chart visually alive '
            'without implying category groupings.'
        ),
        'attributes': [
            ('clean_content → word tokens', 'Categorical — Nominal', '#E8186D'),
            ('word frequency (derived)', 'Quantitative — Discrete', '#14B8A6'),
        ],
        'channels': [
            ('📏 Bar length (X axis)', 'Frequency count encoded as length — the longer the bar, the more times that word appeared.'),
            ('📍 Y position (rank)', 'Words sorted by descending frequency — rank is readable top-to-bottom.'),
            ('🎨 Color spectrum', 'Each word gets a distinct hue from the palette — differentiates bars visually without implying category grouping.'),
            ('📝 Count label', 'Raw count printed outside each bar for precision reading without axis scanning.'),
        ],
    },
    'fig-violin': {
        'title': 'Post Length Distribution',
        'icon': '🎻',
        'accent': '#0EA5E9',
        'description': (
            'Violin plots showing the full distribution of post text lengths (in characters) for each category. '
            'Unlike a box plot, the violin shape reveals the probability density — where character counts are concentrated. '
            'An embedded box (with median line) and mean line are overlaid for precise summary statistics. '
            'Electronics posts show the widest spread, confirming that buyers write both quick questions and detailed research posts.'
        ),
        'attributes': [
            ('text_length', 'Quantitative — Continuous', '#7C3AED'),
            ('category', 'Categorical — Nominal', '#E8186D'),
        ],
        'channels': [
            ('📍 Y position', 'Character count on the Y axis — higher position = longer post.'),
            ('📐 Violin width', 'Width at any Y position encodes probability density — wider = more posts at that length.'),
            ('🎨 Color (hue)', 'Category color fills each violin — consistent with the rest of the dashboard.'),
            ('📦 Embedded box + mean line', 'Box shows IQR + median; dashed mean line reveals skew when median ≠ mean.'),
        ],
    },
    'fig-heatmap': {
        'title': 'Activity Heatmap (Day × Month)',
        'icon': '🌡️',
        'accent': '#FB7185',
        'description': (
            'A two-dimensional heatmap cross-tabulating day of the week (rows) against month (columns). '
            'Each cell\'s color intensity encodes how many posts were published on that day-month combination. '
            'This reveals temporal patterns: are there certain days or months with unusually high posting activity? '
            'The pink-to-dark-rose color scale was chosen to stay within the dashboard\'s editorial palette while '
            'providing enough luminance contrast to read low vs. high cells clearly.'
        ),
        'attributes': [
            ('date → day_name', 'Categorical — Ordinal', '#FF8C42'),
            ('date → month_name', 'Temporal', '#0EA5E9'),
            ('post count (derived)', 'Quantitative — Discrete', '#14B8A6'),
        ],
        'channels': [
            ('📍 X position', 'Month on X axis — temporal order (Jan → Dec) encoded by left-to-right position.'),
            ('📍 Y position', 'Day of week on Y axis — ordinal order (Mon → Sun) encoded by top-to-bottom position.'),
            ('🌡️ Color intensity (luminance)', 'Post count encoded as color darkness — light pink = few posts, dark rose = many posts.'),
            ('📝 Hover tooltip', 'Exact count shown on hover to compensate for the perceptual limits of color-only encoding.'),
        ],
    },
    'fig-knn': {
        'title': 'KNN Accuracy by K',
        'icon': '🤖',
        'accent': '#E8186D',
        'description': (
            'A line chart showing how KNN classification accuracy (%) changes as K (number of neighbors) increases from 1 to N. '
            'The model was trained to classify posts into their product category using only three numerical features: '
            'score, comments, and text_length (after StandardScaler normalization, 80/20 train-test split). '
            'The best result — K=5, 60.47% accuracy — is annotated directly on the chart. '
            'The area fill under the curve emphasizes the magnitude of accuracy, not just its trend.'
        ),
        'attributes': [
            ('K value', 'Quantitative — Discrete', '#14B8A6'),
            ('accuracy (%)', 'Quantitative — Continuous', '#7C3AED'),
        ],
        'channels': [
            ('📍 X position', 'K value on X axis — discrete integer values ordered left-to-right.'),
            ('📍 Y position', 'Accuracy percentage on Y axis — higher = better model performance.'),
            ('📈 Line + area fill', 'Line shows accuracy trend across K values; filled area below reinforces magnitude.'),
            ('⭕ Markers', 'Individual points mark the exact accuracy at each tested K value.'),
            ('📌 Annotation', 'Best K and its accuracy are annotated with an arrow to draw attention to the optimal hyperparameter.'),
        ],
    },
}

# ─── Reddit icon (base64 data URI) ──────────────────────────────────────────
_REDDIT_B64 = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB2aWV3Qm94PSIwIDAgMjAgMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+"
    "PGNpcmNsZSBjeD0iMTAiIGN5PSIxMCIgcj0iMTAiIGZpbGw9IiNGRjQ1MDAiLz48cGF0aCBmaWxsPSJ3"
    "aGl0ZSIgZD0iTTE2LjY3IDEwYTEuNDYgMS40NiAwIDAwLTIuNDctMSA3LjEyIDcuMTIgMCAwMC0zLjg1"
    "LTEuMjNsLjY1LTMuMDggMi4xMy40NWExIDEgMCAxMDEuMDctMSAxIDEgMCAwMC0uOTYuNjhsLTIuMzgt"
    "LjVhLjE2LjE2IDAgMDAtLjE5LjEybC0uNzMgMy40NGE3LjE0IDcuMTQgMCAwMC0zLjg5IDEuMjMgMS40"
    "NiAxLjQ2IDAgMTAtMS42MSAyLjM5IDIuODcgMi44NyAwIDAwMCAuNDRjMCAyLjI0IDIuNjEgNC4wNiA1"
    "LjgzIDQuMDZzNS44My0xLjgyIDUuODMtNC4wNmEyLjg3IDIuODcgMCAwMDAtLjQ0IDEuNDYgMS40NiAw"
    "IDAwLjQ3LTEuNXpNNy4yNyAxMWExIDEgMCAxMTEgMSAxIDEgMCAwMS0xLTF6bTUuNTggMi42NWEzLjQ3"
    "IDMuNDcgMCAwMS0yLjg1LjY2IDMuNDcgMy40NyAwIDAxLTIuODUtLjY2LjE5LjE5IDAgMDEuMjctLjI3"
    "IDMuMTIgMy4xMiAwIDAwMi41OC40OSAzLjEyIDMuMTIgMCAwMDIuNTgtLjQ5LjE5LjE5IDAgMDEuMjcu"
    "Mjd6bS0uMTctMS42NWExIDEgMCAxMTEtMSAxIDEgMCAwMS0xIDF6Ii8+PC9zdmc+"
)
def reddit_icon(size=22):
    return html.Img(src=_REDDIT_B64, style={'width': f'{size}px', 'height': f'{size}px'})

# ─── Plot theme ──────────────────────────────────────────────────────────────
def apply_theme(fig, title=""):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="'Playfair Display','Georgia',serif", color=C['ink'], size=12),
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=13, color=C['ink'], family="'DM Sans',sans-serif"),
            x=0.01, xref='paper', pad=dict(l=6, t=6),
        ),
        margin=dict(l=12, r=12, t=48, b=12),
        legend=dict(
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor=C['border_med'], borderwidth=1,
            font=dict(size=11, color=C['ink_mid'], family="'DM Sans',sans-serif"),
        ),
        xaxis=dict(
            gridcolor=C['grid'], gridwidth=1,
            zerolinecolor=C['border'],
            tickfont=dict(color=C['ink_mid'], size=11, family="'DM Sans',sans-serif"),
            linecolor=C['border_med'],
        ),
        yaxis=dict(
            gridcolor=C['grid'], gridwidth=1,
            zerolinecolor=C['border'],
            tickfont=dict(color=C['ink_mid'], size=11, family="'DM Sans',sans-serif"),
            linecolor=C['border_med'],
        ),
    )
    return fig

# ─── Chart functions ──────────────────────────────────────────────────────────
def fig_donut(filtered):
    counts = filtered['category'].value_counts().reset_index()
    counts.columns = ['category', 'count']
    colors = [CAT_COLORS.get(c, '#ccc') for c in counts['category']]
    fig = go.Figure(go.Pie(
        labels=counts['category'].str.capitalize(), values=counts['count'],
        hole=0.64,
        marker=dict(colors=colors, line=dict(color='white', width=3)),
        textinfo='label+percent',
        textfont=dict(size=12, color=C['ink'], family="'DM Sans',sans-serif"),
        pull=[0.05] * len(counts),
        hovertemplate="<b>%{label}</b><br>Posts: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{len(filtered)}</b><br><span style='font-size:10px;color:{C['ink_mid']}'>posts</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=22, color=C['ink']), align='center',
    )
    return apply_theme(fig, "Category Distribution")

def fig_score_bar(filtered):
    stats = filtered.groupby('category').agg(avg_score=('score','mean')).reset_index()
    stats = stats.sort_values('avg_score', ascending=True)
    colors = [CAT_COLORS.get(c,'#ccc') for c in stats['category']]
    fig = go.Figure(go.Bar(
        x=stats['avg_score'], y=stats['category'].str.capitalize(),
        orientation='h',
        marker=dict(color=colors, opacity=0.88, line=dict(color='rgba(0,0,0,0)')),
        text=[f"  {v:.1f}" for v in stats['avg_score']],
        textposition='outside', textfont=dict(size=13, color=C['ink']),
        hovertemplate="<b>%{y}</b><br>Avg Score: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(showlegend=False, bargap=0.38, xaxis_title="Average Score")
    return apply_theme(fig, "Average Score by Category")

def fig_timeline(filtered):
    monthly = (
        filtered.groupby(['month_num','month_name','category'])
        .size().reset_index(name='count').sort_values('month_num')
    )
    fig = px.line(monthly, x='month_name', y='count', color='category',
        color_discrete_map=CAT_COLORS, markers=True, line_shape='spline')
    fig.update_traces(line=dict(width=2.5), marker=dict(size=8, line=dict(width=2, color='white')))
    fig.update_layout(xaxis_title="Month", yaxis_title="Posts")
    return apply_theme(fig, "Monthly Post Activity")

def fig_scatter(filtered):
    f = filtered.copy()
    f['size_col'] = np.log1p(f['text_length']) * 4 + 6
    fig = px.scatter(f, x='score', y='comments', color='category',
        color_discrete_map=CAT_COLORS, size='size_col', size_max=22, opacity=0.78,
        hover_data={'title': True, 'score': True, 'comments': True, 'size_col': False})
    fig.update_traces(marker=dict(line=dict(width=1.5, color='white')))
    fig.update_xaxes(title="Score (Upvotes)", type='log')
    fig.update_yaxes(title="Comments")
    return apply_theme(fig, "Score vs Comments  (size = text length)")

def fig_keywords(filtered):
    all_text = ' '.join(filtered['clean_content'].dropna())
    words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
    freq = Counter(w for w in words if w not in STOPWORDS)
    top = pd.DataFrame(freq.most_common(12), columns=['word','freq'])
    spectrum = [
        C['magenta'], C['rose'],   C['tangerine'], C['gold'],
        C['mint'],    C['teal'],   C['sky'],        C['violet'],
        C['orchid'],  C['coral'],  C['magenta'],    C['teal'],
    ]
    colors = spectrum[:len(top)]
    fig = go.Figure(go.Bar(
        x=top['freq'], y=top['word'], orientation='h',
        marker=dict(color=colors[::-1], opacity=0.85, line=dict(color='rgba(0,0,0,0)')),
        text=[f"  {v}" for v in top['freq']], textposition='outside',
        textfont=dict(size=12, color=C['ink']),
        hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(autorange='reversed'), xaxis_title="Frequency", showlegend=False)
    return apply_theme(fig, "Top Keywords in Posts")

def fig_violin(filtered):
    fig = go.Figure()
    for cat in CATEGORIES:
        sub = filtered[filtered['category'] == cat]['text_length']
        if len(sub) < 3: continue
        fig.add_trace(go.Violin(
            y=sub, name=cat.capitalize(),
            line_color=CAT_COLORS.get(cat,'#ccc'),
            fillcolor=CAT_COLORS.get(cat,'#ccc'),
            opacity=0.5, box_visible=True, meanline_visible=True, hoverinfo='name+y',
        ))
    fig.update_layout(yaxis_title="Characters", showlegend=False)
    return apply_theme(fig, "Post Length Distribution")

def fig_heatmap(filtered):
    DAY_ORDER = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    heat = filtered.groupby(['day_name','month_name']).size().reset_index(name='count')
    pivot = heat.pivot(index='day_name', columns='month_name', values='count').fillna(0)
    pivot = pivot.reindex([d for d in DAY_ORDER if d in pivot.index])
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[
            [0.0, '#FDE8EF'], [0.3, '#F9A8C9'],
            [0.65, '#E8186D'], [1.0, '#7C0040'],
        ],
        hovertemplate="<b>%{y}</b> · %{x}<br>Posts: %{z}<extra></extra>",
        showscale=True,
        colorbar=dict(
            tickfont=dict(color=C['ink_mid'], size=10),
            thickness=12, len=0.85,
            title=dict(text="Posts", font=dict(color=C['ink_mid'], size=10)),
        ),
    ))
    return apply_theme(fig, "Activity Heatmap  (Day × Month)")

def fig_top_posts(filtered):
    top = filtered.nlargest(8, 'score')
    colors = [CAT_COLORS.get(c,'#ccc') for c in top['category']]
    labels = [t[:48]+'...' if len(t) > 48 else t for t in top['title']]
    fig = go.Figure(go.Bar(
        x=top['score'], y=labels, orientation='h',
        marker=dict(color=colors, opacity=0.88, line=dict(color='rgba(0,0,0,0)')),
        text=[f"  {s}" for s in top['score']], textposition='outside',
        textfont=dict(size=12, color=C['ink']),
        customdata=top[['category','comments']].values,
        hovertemplate="<b>%{y}</b><br>Category: %{customdata[0]}<br>Score: %{x}<br>Comments: %{customdata[1]}<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(autorange='reversed'), showlegend=False, xaxis_title="Score")
    return apply_theme(fig, "Top 8 Posts by Score")

def fig_knn():
    best_idx = knn_df['accuracy'].idxmax()
    best_k   = knn_df.loc[best_idx, 'K']
    best_acc = knn_df.loc[best_idx, 'accuracy']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=knn_df['K'], y=knn_df['accuracy'],
        mode='lines+markers',
        line=dict(color=C['magenta'], width=3),
        marker=dict(size=9, color=C['orchid'], line=dict(width=2, color='white')),
        fill='tozeroy', fillcolor='rgba(232,24,109,0.08)',
        hovertemplate="K=%{x}<br>Accuracy: %{y:.2f}%<extra></extra>",
    ))
    fig.add_annotation(
        x=best_k, y=best_acc,
        text=f"<b>Best K={int(best_k)}<br>{best_acc:.1f}%</b>",
        showarrow=True, arrowhead=2, arrowcolor=C['magenta'],
        font=dict(color=C['ink'], size=11),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor=C['border_med'], borderwidth=1,
    )
    fig.update_layout(xaxis_title="K value", yaxis_title="Accuracy (%)", showlegend=False)
    return apply_theme(fig, "KNN Accuracy by K")

# ─── KPI Card ────────────────────────────────────────────────────────────────
def kpi_card(label, value, icon, accent):
    return dbc.Col(
        html.Div([
            html.Div(icon, style={
                'fontSize': '28px', 'marginBottom': '10px',
                'filter': 'drop-shadow(0 2px 6px rgba(0,0,0,0.12))',
            }),
            html.Div(value, style={
                'fontSize': '36px', 'fontWeight': '900',
                'color': accent, 'lineHeight': '1',
                'letterSpacing': '-1.5px',
                'fontFamily': "'Playfair Display', serif",
            }),
            html.Div(label, style={
                'fontSize': '9px', 'fontWeight': '700',
                'color': C['ink_soft'], 'marginTop': '6px',
                'textTransform': 'uppercase', 'letterSpacing': '0.14em',
                'fontFamily': "'DM Sans', sans-serif",
            }),
            html.Div(style={
                'height': '3px', 'width': '32px',
                'background': accent,
                'borderRadius': '2px', 'marginTop': '10px',
                'opacity': '0.7',
            }),
        ], style={
            'background': 'rgba(255,255,255,0.75)',
            'backdropFilter': 'blur(12px)',
            'WebkitBackdropFilter': 'blur(12px)',
            'border': '1.5px solid rgba(255,255,255,0.9)',
            'borderRadius': '20px',
            'padding': '22px 20px',
            'boxShadow': '0 8px 32px rgba(232,24,109,0.10), inset 0 1px 0 rgba(255,255,255,0.8)',
            'position': 'relative',
            'overflow': 'hidden',
        }),
        xs=6, sm=6, md=3,
        style={'marginBottom': '16px'},
    )

CARD = {
    'background': 'rgba(255,255,255,0.70)',
    'backdropFilter': 'blur(14px)',
    'WebkitBackdropFilter': 'blur(14px)',
    'border': '1.5px solid rgba(255,255,255,0.90)',
    'borderRadius': '20px',
    'padding': '6px',
    'marginBottom': '16px',
    'boxShadow': '0 4px 24px rgba(232,24,109,0.08)',
    'position': 'relative',
}

# ─── Click-to-expand chart cards ─────────────────────────────────────────────
def chart_card_with_hint(graph_id, height='340px'):
    """
    Wraps a dcc.Graph in a clickable card.
    n_clicks on the outer Div triggers the modal — no clickData needed.
    The graph itself has pointer-events disabled so clicks always hit the Div.
    """
    return html.Div([
        # Expand hint badge (CSS shows it on .chart-card:hover)
        html.Div([
            html.Span("⛶ ", style={'fontSize': '13px'}),
            html.Span("expand", style={'fontSize': '10px', 'fontWeight': '700',
                'letterSpacing': '0.08em', 'textTransform': 'uppercase'}),
        ], className='expand-hint'),
        # Graph — pointer-events:none so clicks bubble up to the Div
        dcc.Graph(
            id=graph_id,
            config={'displayModeBar': False},
            style={'height': height, 'pointerEvents': 'none'},
        ),
    ], style={
        **CARD,
        'cursor': 'pointer',
        'transition': 'transform 0.2s ease, box-shadow 0.2s ease',
        'position': 'relative',
    }, className='chart-card', id=f'card-{graph_id}', n_clicks=0)

# ─── Modal builder ───────────────────────────────────────────────────────────
def build_modal():
    """
    Single modal reused for all charts.
    Content is populated dynamically via callback.
    """
    return html.Div([
        # Backdrop
        html.Div(id='modal-backdrop', n_clicks=0, style={
            'display': 'none',
            'position': 'fixed', 'inset': '0',
            'background': 'rgba(42,17,32,0.55)',
            'backdropFilter': 'blur(6px)',
            'WebkitBackdropFilter': 'blur(6px)',
            'zIndex': '9000',
        }),
        # Modal panel
        html.Div([
            # Close button
            html.Div([
                html.Button("✕", id='modal-close', style={
                    'background': 'rgba(255,255,255,0.90)',
                    'border': f'1.5px solid {C["border_med"]}',
                    'borderRadius': '12px',
                    'padding': '6px 14px',
                    'fontSize': '16px', 'fontWeight': '700',
                    'color': C['ink_mid'], 'cursor': 'pointer',
                    'fontFamily': "'DM Sans', sans-serif",
                    'lineHeight': '1',
                }),
            ], style={'position': 'absolute', 'top': '16px', 'right': '16px', 'zIndex': '10'}),

            # Dynamic content area
            html.Div(id='modal-content'),

        ], id='modal-panel', style={
            'display': 'none',
            'position': 'fixed',
            'top': '50%', 'left': '50%',
            'transform': 'translate(-50%, -50%) scale(0.96)',
            'width': 'min(1100px, 96vw)',
            'maxHeight': '90vh',
            'overflowY': 'auto',
            'background': 'rgba(255,240,246,0.97)',
            'backdropFilter': 'blur(24px)',
            'WebkitBackdropFilter': 'blur(24px)',
            'border': '2px solid rgba(255,255,255,0.95)',
            'borderRadius': '28px',
            'padding': '32px',
            'boxShadow': '0 32px 80px rgba(42,17,32,0.22), inset 0 1px 0 rgba(255,255,255,0.9)',
            'zIndex': '9001',
            'transition': 'transform 0.25s ease, opacity 0.25s ease',
        }),
    ])

def build_modal_content(chart_id, figure):
    """Build the modal inner content given a chart_id and its current figure."""
    meta = CHART_META.get(chart_id, {})
    if not meta:
        return html.Div("No information available.")

    accent = meta.get('accent', C['magenta'])

    # ── Attribute pills ──────────────────────────────────────────────────────
    TYPE_COLORS = {
        'Categorical — Nominal':     C['magenta'],
        'Categorical (Nominal)':     C['magenta'],
        'Categorical — Ordinal':     C['tangerine'],
        'Categorical (Ordinal)':     C['tangerine'],
        'Quantitative — Discrete':   C['teal'],
        'Quantitative (Discrete)':   C['teal'],
        'Quantitative — Continuous': C['violet'],
        'Quantitative (Continuous)': C['violet'],
        'Temporal':                  C['sky'],
        'Quantitive (Temporal)':     C['sky'],
    }

    def attr_pill(attr_name, attr_type, attr_color):
        tc = TYPE_COLORS.get(attr_type, attr_color)
        return html.Div([
            html.Div(attr_name, style={
                'fontFamily': "'Courier New', monospace",
                'fontSize': '12px', 'fontWeight': '700',
                'color': tc, 'marginBottom': '4px',
            }),
            html.Div(attr_type, style={
                'fontSize': '10px', 'fontWeight': '700',
                'color': C['ink_soft'],
                'textTransform': 'uppercase', 'letterSpacing': '0.08em',
                'fontFamily': "'DM Sans', sans-serif",
            }),
        ], style={
            'background': f'{tc}10',
            'border': f'1.5px solid {tc}33',
            'borderRadius': '12px',
            'padding': '10px 14px',
            'marginRight': '8px', 'marginBottom': '8px',
            'display': 'inline-block',
        })

    # ── Channel rows ─────────────────────────────────────────────────────────
    def channel_row(channel_label, channel_desc):
        parts = channel_label.split(' ', 1)
        icon = parts[0] if parts else '•'
        label = parts[1] if len(parts) > 1 else channel_label
        return html.Div([
            html.Div(icon, style={
                'fontSize': '18px', 'width': '36px', 'height': '36px',
                'background': f'{accent}12',
                'border': f'1.5px solid {accent}30',
                'borderRadius': '10px',
                'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
                'flexShrink': '0', 'marginRight': '12px',
            }),
            html.Div([
                html.Div(label, style={
                    'fontSize': '12px', 'fontWeight': '700', 'color': C['ink'],
                    'fontFamily': "'DM Sans', sans-serif", 'marginBottom': '2px',
                }),
                html.Div(channel_desc, style={
                    'fontSize': '11.5px', 'color': C['ink_mid'], 'lineHeight': '1.65',
                    'fontFamily': "'DM Sans', sans-serif",
                }),
            ]),
        ], style={
            'display': 'flex', 'alignItems': 'flex-start',
            'background': 'rgba(255,255,255,0.65)',
            'border': '1.5px solid rgba(255,255,255,0.9)',
            'borderLeft': f'3px solid {accent}',
            'borderRadius': '12px',
            'padding': '12px 14px',
            'marginBottom': '8px',
        })

    # ── Enlarged figure (same figure, bigger margins) ────────────────────────
    if figure:
        big_fig = go.Figure(figure)
        big_fig.update_layout(
            margin=dict(l=20, r=20, t=56, b=20),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        enlarged_graph = dcc.Graph(
            figure=big_fig,
            config={'displayModeBar': True, 'modeBarButtonsToRemove': ['lasso2d','select2d']},
            style={'height': '400px'},
        )
    else:
        enlarged_graph = html.Div()

    return html.Div([
        # Header
        html.Div([
            html.Div([
                html.Div(meta.get('icon','📊'), style={
                    'fontSize': '28px', 'width': '56px', 'height': '56px',
                    'background': f'{accent}15', 'border': f'2px solid {accent}33',
                    'borderRadius': '16px', 'display': 'flex',
                    'alignItems': 'center', 'justifyContent': 'center',
                    'marginRight': '18px', 'flexShrink': '0',
                }),
                html.Div([
                    html.Div(meta.get('title', ''), style={
                        'fontFamily': "'Playfair Display', serif",
                        'fontSize': '22px', 'fontWeight': '900', 'color': C['ink'],
                        'marginBottom': '4px',
                    }),
                    html.Div("Chart Overview", style={
                        'fontSize': '10px', 'fontWeight': '800', 'color': accent,
                        'textTransform': 'uppercase', 'letterSpacing': '0.18em',
                        'fontFamily': "'DM Sans', sans-serif",
                    }),
                ]),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),

            # Divider
            html.Div(style={
                'height': '1.5px',
                'background': f'linear-gradient(90deg, {accent}55, transparent)',
                'marginBottom': '20px',
            }),

            # Enlarged chart
            html.Div(enlarged_graph, style={
                'background': 'rgba(255,255,255,0.60)',
                'border': f'1.5px solid {accent}22',
                'borderRadius': '16px',
                'overflow': 'hidden',
                'marginBottom': '24px',
            }),

            # Description
            html.Div([
                html.Div("✦  WHAT THIS CHART SHOWS", style={
                    'fontSize': '10px', 'fontWeight': '800', 'color': C['ink_soft'],
                    'letterSpacing': '0.20em', 'marginBottom': '10px',
                    'fontFamily': "'DM Sans', sans-serif",
                }),
                html.Div(meta.get('description', ''), style={
                    'fontSize': '13px', 'color': C['ink_mid'], 'lineHeight': '1.80',
                    'fontFamily': "'DM Sans', sans-serif",
                }),
            ], style={
                'background': 'rgba(255,255,255,0.60)',
                'border': '1.5px solid rgba(255,255,255,0.9)',
                'borderRadius': '16px', 'padding': '18px 20px',
                'marginBottom': '20px',
            }),

            # Attributes used
            html.Div([
                html.Div("✦  ATTRIBUTES USED", style={
                    'fontSize': '10px', 'fontWeight': '800', 'color': C['ink_soft'],
                    'letterSpacing': '0.20em', 'marginBottom': '12px',
                    'fontFamily': "'DM Sans', sans-serif",
                }),
                html.Div([
                    attr_pill(a, t, c)
                    for a, t, c in meta.get('attributes', [])
                ]),
            ], style={'marginBottom': '20px'}),

            # Visual channels
            html.Div([
                html.Div("✦  VISUAL CHANNELS", style={
                    'fontSize': '10px', 'fontWeight': '800', 'color': C['ink_soft'],
                    'letterSpacing': '0.20em', 'marginBottom': '12px',
                    'fontFamily': "'DM Sans', sans-serif",
                }),
                html.Div([
                    channel_row(ch_label, ch_desc)
                    for ch_label, ch_desc in meta.get('channels', [])
                ]),
            ]),
        ]),
    ])

# ─── Dataset Intro ────────────────────────────────────────────────────────────
def dataset_intro():
    def pill(text, color):
        return html.Span(text, style={
            'background': f'{color}18', 'color': color,
            'border': f'1.5px solid {color}55',
            'borderRadius': '20px', 'padding': '4px 13px',
            'fontSize': '11px', 'fontWeight': '700',
            'marginRight': '8px', 'marginBottom': '7px',
            'display': 'inline-block',
            'fontFamily': "'DM Sans', sans-serif",
            'letterSpacing': '0.02em',
        })

    def stat_block(num, label, accent):
        return html.Div([
            html.Div(num, style={
                'fontSize': '26px', 'fontWeight': '900', 'color': accent,
                'lineHeight': '1', 'fontFamily': "'Playfair Display', serif",
                'letterSpacing': '-1px',
            }),
            html.Div(label, style={
                'fontSize': '9px', 'color': C['ink_soft'], 'marginTop': '5px',
                'textTransform': 'uppercase', 'letterSpacing': '0.12em',
                'fontFamily': "'DM Sans', sans-serif", 'fontWeight': '700',
            }),
            html.Div(style={'height':'2px','width':'24px','background':accent,'borderRadius':'2px','marginTop':'8px'}),
        ], style={
            'background': 'rgba(255,255,255,0.65)',
            'backdropFilter': 'blur(10px)',
            'border': '1.5px solid rgba(255,255,255,0.9)',
            'borderRadius': '16px',
            'padding': '18px 20px',
            'textAlign': 'left',
            'flex': '1',
            'marginRight': '10px',
            'boxShadow': '0 4px 16px rgba(232,24,109,0.07)',
        })

    def step_card(num, title, desc, accent):
        return html.Div([
            html.Div([
                html.Span(num, style={
                    'fontSize': '11px', 'fontWeight': '800', 'color': accent,
                    'fontFamily': "'DM Sans', sans-serif", 'letterSpacing': '0.06em',
                }),
            ], style={
                'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                'background': f'{accent}18', 'border': f'1.5px solid {accent}44',
                'borderRadius': '8px', 'padding': '4px 10px',
                'marginBottom': '12px',
            }),
            html.Div(title, style={
                'fontSize': '13px', 'fontWeight': '700', 'color': C['ink'],
                'marginBottom': '7px', 'fontFamily': "'DM Sans', sans-serif",
            }),
            html.Div(desc, style={
                'fontSize': '11.5px', 'color': C['ink_mid'], 'lineHeight': '1.7',
                'fontFamily': "'DM Sans', sans-serif",
            }),
        ], style={
            'background': 'rgba(255,255,255,0.65)',
            'backdropFilter': 'blur(10px)',
            'border': f'1.5px solid rgba(255,255,255,0.9)',
            'borderTop': f'3px solid {accent}',
            'borderRadius': '16px',
            'padding': '18px',
            'flex': '1',
            'marginRight': '10px',
            'boxShadow': '0 4px 14px rgba(0,0,0,0.05)',
        })

    def insight_card(icon, title, desc, accent):
        return html.Div([
            html.Div([
                html.Div(icon, style={
                    'fontSize': '20px',
                    'width': '38px', 'height': '38px',
                    'background': f'{accent}18',
                    'border': f'1.5px solid {accent}33',
                    'borderRadius': '10px',
                    'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
                    'marginRight': '12px', 'flexShrink': '0',
                }),
                html.Div(title, style={
                    'fontSize': '13px', 'fontWeight': '700', 'color': C['ink'],
                    'fontFamily': "'DM Sans', sans-serif",
                }),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'}),
            html.Div(desc, style={
                'fontSize': '11.5px', 'color': C['ink_mid'],
                'lineHeight': '1.65', 'fontFamily': "'DM Sans', sans-serif",
            }),
        ], style={
            'background': 'rgba(255,255,255,0.65)',
            'backdropFilter': 'blur(10px)',
            'border': '1.5px solid rgba(255,255,255,0.9)',
            'borderRadius': '16px',
            'padding': '16px',
            'boxShadow': '0 4px 14px rgba(0,0,0,0.04)',
        })

    return html.Div([
        html.Div([
            html.Div([
                html.Div("✦  ABOUT THIS PROJECT", style={
                    'fontSize': '10px', 'fontWeight': '800', 'color': C['magenta'],
                    'letterSpacing': '0.22em', 'marginBottom': '12px',
                    'fontFamily': "'DM Sans', sans-serif",
                }),
                html.Div([
                    html.Span("Algerian ", style={
                        'fontFamily': "'Playfair Display', serif",
                        'fontSize': '32px', 'fontWeight': '900',
                        'fontStyle': 'italic', 'color': C['magenta'],
                    }),
                    html.Span("Product Reviews", style={
                        'fontFamily': "'Playfair Display', serif",
                        'fontSize': '32px', 'fontWeight': '900', 'color': C['ink'],
                    }),
                ]),
                html.Div(
                    "A scraped & analysed dataset of consumer opinions from Algerian Reddit "
                    "communities, covering home appliances, food, electronics, and cosmetics. "
                    "Built end-to-end with custom keyword scraping, text cleaning, and KNN classification.",
                    style={
                        'fontSize': '13px', 'color': C['ink_mid'], 'lineHeight': '1.75',
                        'maxWidth': '560px', 'marginTop': '12px',
                        'fontFamily': "'DM Sans', sans-serif",
                    }
                ),
                html.Div([
                    pill("r/AlgeriaRates", C['magenta']),
                    pill("r/Algeria",      C['tangerine']),
                    pill("April 2026",     C['teal']),
                    pill("Python · pandas · sklearn", C['violet']),
                ], style={'marginTop': '16px'}),
            ], style={'flex': '1', 'paddingRight': '30px', 'minWidth': '260px'}),

            html.Div([
                stat_block("430",   "Reviews",      C['magenta']),
                stat_block("4",     "Categories",   C['tangerine']),
                stat_block("60.5%", "KNN Accuracy", C['teal']),
                stat_block("K = 5", "Best K",       C['violet']),
            ], style={'display':'flex','flexWrap':'wrap','marginRight':'-10px','gap':'0','alignItems':'stretch'}),
        ], style={
            'display': 'flex', 'alignItems': 'flex-start',
            'flexWrap': 'wrap', 'gap': '24px', 'marginBottom': '28px',
        }),

        html.Div([
            html.Div("✦  HOW IT WAS BUILT", style={
                'fontSize': '10px', 'fontWeight': '800', 'color': C['ink_soft'],
                'letterSpacing': '0.22em', 'marginBottom': '14px',
                'fontFamily': "'DM Sans', sans-serif",
            }),
            html.Div([
                step_card("01", "Data Collection",
                    "Scraped 500+ posts from r/AlgeriaRates & r/Algeria via the Reddit JSON API with retry logic and 80+ product keywords (Condor, Samsung, Rouiba...).",
                    C['magenta']),
                step_card("02", "Data Cleaning",
                    "Removed duplicates, handled missing values, cleaned text, stripped English & French stopwords and filler review words.",
                    C['tangerine']),
                step_card("03", "Analysis",
                    "Statistical & frequency analysis (Counter). Category breakdowns on scores, comments, text length, and brand mentions.",
                    C['teal']),
                step_card("04", "ML — KNN",
                    "KNN trained on score, comments & text_length (StandardScaler, 80/20 split). Best accuracy at K=5: 60.47%.",
                    C['violet']),
            ], style={'display':'flex','flexWrap':'wrap','marginRight':'-10px'}),
        ], style={'marginBottom': '28px'}),

        html.Div("✦  KEY FINDINGS", style={
            'fontSize': '10px', 'fontWeight': '800', 'color': C['ink_soft'],
            'letterSpacing': '0.22em', 'marginBottom': '14px',
            'fontFamily': "'DM Sans', sans-serif",
        }),
        dbc.Row([
            dbc.Col(insight_card("🏠", "Appliances Dominate",
                "62.1% of posts discuss home appliances: fridges, washing machines & ACs are the most talked-about.", C['magenta']),
                md=4, style={'marginBottom': '10px'}),
            dbc.Col(insight_card("🍽️", "Food Gets the Most Upvotes",
                "Food posts earn the highest avg score (23.0); quality & availability resonate widely.", C['tangerine']),
                md=4, style={'marginBottom': '10px'}),
            dbc.Col(insight_card("📝", "Electronics = Longest Reviews",
                "Electronics posts average 588 characters which indicates buyers research thoroughly before purchasing.", C['teal']),
                md=4, style={'marginBottom': '10px'}),
        ]),
        dbc.Row([
            dbc.Col(insight_card("🏷️", "LG Leads Brand Mentions",
                "LG appears 462 times vs. only 3 for Samsung: it is deeply embedded in Algerian appliance culture.", C['violet']),
                md=4, style={'marginBottom': '4px'}),
            dbc.Col(insight_card("💬", "High Comment Engagement",
                "Appliances (20.8 avg) and Electronics (20.2 avg) generate the most discussion.", C['orchid']),
                md=4, style={'marginBottom': '4px'}),
            dbc.Col(insight_card("🤖", "Promising KNN Baseline",
                "With only 3 simple features, KNN reached 60.47% , a solid baseline for richer future models.", C['coral']),
                md=4, style={'marginBottom': '4px'}),
        ]),

    ], style={
        'background': 'rgba(255,255,255,0.50)',
        'backdropFilter': 'blur(20px)',
        'WebkitBackdropFilter': 'blur(20px)',
        'border': '2px solid rgba(255,255,255,0.85)',
        'borderRadius': '24px',
        'padding': '30px 28px',
        'marginBottom': '24px',
        'boxShadow': '0 8px 40px rgba(232,24,109,0.09), inset 0 1px 0 rgba(255,255,255,0.8)',
        'position': 'relative',
        'overflow': 'hidden',
    })


# ─── Data Documentation Panel ─────────────────────────────────────────────────
def data_documentation():
    def section_label(text):
        return html.Div(text, style={
            'fontSize': '10px', 'fontWeight': '800', 'color': C['ink_soft'],
            'letterSpacing': '0.22em', 'marginBottom': '16px',
            'textTransform': 'uppercase', 'fontFamily': "'DM Sans', sans-serif",
        })

    items_block = html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Div("📄", style={
                        'fontSize': '24px', 'width': '48px', 'height': '48px',
                        'background': f'{C["magenta"]}12', 'border': f'1.5px solid {C["magenta"]}30',
                        'borderRadius': '14px', 'display': 'flex',
                        'alignItems': 'center', 'justifyContent': 'center',
                        'marginRight': '16px', 'flexShrink': '0',
                    }),
                    html.Div([
                        html.Div("The Items", style={
                            'fontSize': '14px', 'fontWeight': '800', 'color': C['ink'],
                            'fontFamily': "'DM Sans', sans-serif", 'marginBottom': '5px',
                        }),
                        html.Div(
                            "Each item in this dataset is a Reddit post — a single consumer opinion "
                            "or question published by an Algerian user on r/AlgeriaRates or r/Algeria. "
                            "Each post represents one real-world review event: a person sharing their "
                            "experience with, or asking about, a specific Algerian product.",
                            style={
                                'fontSize': '12px', 'color': C['ink_mid'], 'lineHeight': '1.75',
                                'fontFamily': "'DM Sans', sans-serif",
                            }
                        ),
                    ]),
                ], style={'display': 'flex', 'alignItems': 'flex-start', 'marginBottom': '16px'}),

                html.Div([
                    html.Div([
                        html.Span("Type of item", style={
                            'fontSize': '9px', 'fontWeight': '800', 'color': C['ink_soft'],
                            'textTransform': 'uppercase', 'letterSpacing': '0.12em',
                            'display': 'block', 'marginBottom': '4px',
                            'fontFamily': "'DM Sans', sans-serif",
                        }),
                        html.Span("Document / Event", style={
                            'fontSize': '13px', 'fontWeight': '700', 'color': C['magenta'],
                            'fontFamily': "'Playfair Display', serif",
                        }),
                    ], style={
                        'flex': '1', 'background': f'{C["magenta"]}08',
                        'border': f'1px solid {C["magenta"]}22',
                        'borderRadius': '12px', 'padding': '14px 16px',
                    }),
                    html.Div([
                        html.Span("Total items", style={
                            'fontSize': '9px', 'fontWeight': '800', 'color': C['ink_soft'],
                            'textTransform': 'uppercase', 'letterSpacing': '0.12em',
                            'display': 'block', 'marginBottom': '4px',
                            'fontFamily': "'DM Sans', sans-serif",
                        }),
                        html.Span("430 posts", style={
                            'fontSize': '13px', 'fontWeight': '700', 'color': C['tangerine'],
                            'fontFamily': "'Playfair Display', serif",
                        }),
                    ], style={
                        'flex': '1', 'background': f'{C["tangerine"]}08',
                        'border': f'1px solid {C["tangerine"]}22',
                        'borderRadius': '12px', 'padding': '14px 16px',
                        'marginLeft': '10px',
                    }),
                    html.Div([
                        html.Span("What they represent", style={
                            'fontSize': '9px', 'fontWeight': '800', 'color': C['ink_soft'],
                            'textTransform': 'uppercase', 'letterSpacing': '0.12em',
                            'display': 'block', 'marginBottom': '4px',
                            'fontFamily': "'DM Sans', sans-serif",
                        }),
                        html.Span("Consumer opinions on Algerian products", style={
                            'fontSize': '12px', 'fontWeight': '700', 'color': C['teal'],
                            'fontFamily': "'Playfair Display', serif",
                        }),
                    ], style={
                        'flex': '2', 'background': f'{C["teal"]}08',
                        'border': f'1px solid {C["teal"]}22',
                        'borderRadius': '12px', 'padding': '14px 16px',
                        'marginLeft': '10px',
                    }),
                ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '0'}),
            ], style={
                'background': 'rgba(255,255,255,0.60)',
                'border': f'1.5px solid {C["border_med"]}',
                'borderRadius': '16px', 'padding': '20px 22px',
                'marginBottom': '16px',
            }),
        ]),
    ], style={'marginBottom': '24px'})

    TYPE_BADGE = {
        'Categorical (Nominal)':     (C['magenta'],   '#FFF0F5'),
        'Categorical (Ordinal)':     (C['tangerine'],  '#FFF5EE'),
        'Quantitative (Discrete)':   (C['teal'],       '#F0FAFA'),
        'Quantitative  (Continuous)': (C['violet'],     '#F5F0FF'),
        'Quantitive  (Temporal)':                  (C['sky'],         '#F0F8FF'),
    }

    def type_badge(label):
        color, bg = TYPE_BADGE.get(label, (C['ink_soft'], '#F5F5F5'))
        return html.Span(label, style={
            'background': bg, 'color': color,
            'border': f'1.5px solid {color}44',
            'borderRadius': '20px', 'padding': '3px 10px',
            'fontSize': '10px', 'fontWeight': '700',
            'fontFamily': "'DM Sans', sans-serif",
            'whiteSpace': 'nowrap',
        })

    ATTRS = [
        ("title", "The headline text of the post.", "Categorical (Nominal)",
         "Free-text string; no inherent order or magnitude between titles."),
        ("category", "Product domain: appliances, food, electronics, or cosmetics.", "Categorical (Nominal)",
         "4 distinct, unordered groups assigned via keyword matching."),
        ("subreddit", "Community: r/AlgeriaRates or r/Algeria.", "Categorical (Nominal)",
         "Two unordered source labels. Neither subreddit is 'higher' than the other."),
        ("score", "Net upvote count — measure of popularity.", "Quantitative (Discrete)",
         "Whole integers (0, 1, 2, …). Ratio scale: score 20 is twice as popular as 10."),
        ("comments", "Number of replies — measure of engagement depth.", "Quantitative (Discrete)",
         "Integer comment counts ≥ 0. Cannot have half a comment."),
        ("text_length", "Character count of the post body.", "Quantitative  (Continuous)",
         "Any non-negative integer value; no natural gap between values."),
        ("date", "Timestamp of publication on Reddit.", "Quantitive  (Temporal)",
         "Parsed into month and day-of-week to expose activity patterns."),
        ("clean_content", "Cleaned body text after stopword removal.", "Categorical (Nominal)",
         "Free-text; each post has a unique string. Source for keyword frequency counting."),
    ]

    header_style = {
        'fontSize': '10px', 'fontWeight': '800', 'color': C['ink_soft'],
        'textTransform': 'uppercase', 'letterSpacing': '0.12em',
        'padding': '10px 14px', 'borderBottom': f'1px solid {C["border_med"]}',
        'fontFamily': "'DM Sans', sans-serif",
        'background': 'rgba(232,24,109,0.04)',
    }
    cell_style = {
        'padding': '11px 14px', 'fontSize': '11.5px', 'color': C['ink'],
        'borderBottom': f'1px solid {C["border"]}',
        'fontFamily': "'DM Sans', sans-serif", 'verticalAlign': 'top',
    }
    mono_style = {**cell_style, 'fontFamily': "'Courier New', monospace",
                  'fontWeight': '700', 'color': C['magenta'], 'fontSize': '12px'}

    attr_table = html.Div([
        html.Table([
            html.Thead(html.Tr([
                html.Th("Attribute",         style={**header_style, 'width': '12%'}),
                html.Th("What it describes", style={**header_style, 'width': '28%'}),
                html.Th("Type",              style={**header_style, 'width': '22%'}),
                html.Th("Justification",     style={**header_style, 'width': '38%'}),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(attr, style=mono_style),
                    html.Td(desc, style={**cell_style, 'fontStyle': 'italic', 'color': C['ink_mid']}),
                    html.Td(type_badge(typ), style={**cell_style, 'paddingTop': '9px'}),
                    html.Td(just, style={**cell_style, 'color': C['ink_mid'], 'lineHeight': '1.65'}),
                ], style={'background': 'rgba(255,255,255,0)' if i % 2 == 0 else 'rgba(232,24,109,0.025)'})
                for i, (attr, desc, typ, just) in enumerate(ATTRS)
            ]),
        ], style={'width': '100%', 'borderCollapse': 'collapse'}),
    ], style={
        'background': 'rgba(255,255,255,0.60)',
        'border': f'1.5px solid {C["border_med"]}',
        'borderRadius': '16px', 'overflow': 'hidden',
        'marginBottom': '24px',
    })

    MAPPINGS = [
        ("🎨", "Color",    C['magenta'],   "category → hue",
         "Color separates the 4 product categories. No category is ranked – each has a fixed, distinct color."),
        ("📍", "Position X/Y", C['tangerine'], "score, comments → axes",
         "Position along the X/Y axis is most accurate for comparing numbers."),
        ("⭕", "Size",     C['teal'],      "text_length → bubble area",
         "Text length (character count) is encoded as bubble area. Larger bubbles = longer posts."),
        ("🌡️", "Color Intensity", C['violet'], "post count → heatmap luminance",
         "In the Day × Month heatmap, luminance encodes post count."),
        ("📏", "Bar Length", C['sky'],    "avg score, frequency → bar length",
         "Bar length directly represents numeric values. One of the most accurate visual channels."),
        ("📈", "Line + Markers", C['orchid'], "time → X position, trend → line slope",
         "Months on the X axis encode temporal order. Line slope reveals trends."),
    ]

    def channel_card(icon, channel, accent, mapping, rationale):
        return html.Div([
            html.Div([
                html.Div(icon, style={
                    'fontSize': '22px', 'width': '44px', 'height': '44px',
                    'background': f'{accent}15', 'border': f'1.5px solid {accent}33',
                    'borderRadius': '12px', 'display': 'flex',
                    'alignItems': 'center', 'justifyContent': 'center',
                    'marginRight': '12px', 'flexShrink': '0',
                }),
                html.Div([
                    html.Div(channel, style={
                        'fontSize': '13px', 'fontWeight': '800', 'color': C['ink'],
                        'fontFamily': "'DM Sans', sans-serif",
                    }),
                    html.Div(mapping, style={
                        'fontSize': '11px', 'color': accent, 'fontWeight': '700',
                        'fontFamily': "'Courier New', monospace", 'marginTop': '2px',
                    }),
                ]),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
            html.Div(rationale, style={
                'fontSize': '11.5px', 'color': C['ink_mid'], 'lineHeight': '1.65',
                'fontFamily': "'DM Sans', sans-serif",
            }),
            html.Div(style={
                'height': '2px', 'background': f'linear-gradient(90deg, {accent}55, transparent)',
                'borderRadius': '2px', 'marginTop': '12px',
            }),
        ], style={
            'background': 'rgba(255,255,255,0.65)',
            'backdropFilter': 'blur(10px)',
            'border': '1.5px solid rgba(255,255,255,0.9)',
            'borderLeft': f'3px solid {accent}',
            'borderRadius': '16px',
            'padding': '18px',
            'boxShadow': '0 4px 14px rgba(0,0,0,0.04)',
        })

    channel_grid = dbc.Row([
        dbc.Col(channel_card(*m), md=4, style={'marginBottom': '12px'})
        for m in MAPPINGS
    ], style={'marginBottom': '24px'})

    def layout_item(num, title, desc, accent):
        return html.Div([
            html.Div(num, style={
                'fontSize': '11px', 'fontWeight': '900', 'color': accent,
                'fontFamily': "'Playfair Display', serif",
                'width': '28px', 'height': '28px',
                'background': f'{accent}15', 'border': f'1.5px solid {accent}44',
                'borderRadius': '8px',
                'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
                'flexShrink': '0', 'marginRight': '14px',
            }),
            html.Div([
                html.Div(title, style={
                    'fontSize': '12px', 'fontWeight': '700', 'color': C['ink'],
                    'fontFamily': "'DM Sans', sans-serif", 'marginBottom': '4px',
                }),
                html.Div(desc, style={
                    'fontSize': '11.5px', 'color': C['ink_mid'], 'lineHeight': '1.65',
                    'fontFamily': "'DM Sans', sans-serif",
                }),
            ]),
        ], style={'display': 'flex', 'alignItems': 'flex-start', 'marginBottom': '16px'})

    LAYOUT_REASONS = [
        ("1", "Sticky header: orientation at all times",
         "The navigation bar stays fixed on top. Users always know the project context while scrolling.", C['magenta']),
        ("2", "About panel first: context before data",
         "The dataset description and key findings come before any chart. This gives readers the background they need to interpret the visualizations.", C['tangerine']),
        ("3", "Data documentation second: transparency",
         "Attribute types and visual channel mappings are placed right after the intro. This makes it easy to check design decisions before exploring the interactive charts.", C['teal']),
        ("4", "Filter bar above all charts: global control",
         "A single filter bar (category, score, subreddit) affects every chart at once. Placing it above all sections avoids duplicate controls and clearly connects filters to updates.", C['violet']),
        ("5", "KPI row: quick numeric summary",
         "Four large-number cards (posts, avg score, avg comments, avg length) give an instant overview before diving into detailed charts.", C['orchid']),
        ("6", "Thematic grouping: natural story flow",
         "Charts are grouped by analytical task: Distribution → Activity → Text → Patterns/ML. This mirrors a logical narrative: 'what', 'when', 'what do people say', 'can a model learn it'.", C['sky']),
        ("7", "Wide + narrow column pairing",
         "Each row pairs a wider chart with a smaller one. This creates visual rhythm, saves space, and lets related charts share the same row without competition.", C['coral']),
    ]

    layout_panel = html.Div([
        html.Div([layout_item(*r) for r in LAYOUT_REASONS]),
    ], style={
        'background': 'rgba(255,255,255,0.60)',
        'border': f'1.5px solid {C["border_med"]}',
        'borderRadius': '16px', 'padding': '22px 24px',
    })

    return html.Div([
        html.Div("✦  ABOUT THIS PROJECT", style={
            'fontSize': '10px', 'fontWeight': '800', 'color': C['magenta'],
            'letterSpacing': '0.22em', 'marginBottom': '6px',
            'fontFamily': "'DM Sans', sans-serif",
        }),
        html.Div([
            html.Span("Data ", style={
                'fontFamily': "'Playfair Display', serif",
                'fontSize': '26px', 'fontWeight': '900', 'fontStyle': 'italic',
                'color': C['magenta'],
            }),
            html.Span("Documentation", style={
                'fontFamily': "'Playfair Display', serif",
                'fontSize': '26px', 'fontWeight': '900', 'color': C['ink'],
            }),
        ], style={'marginBottom': '6px'}),
        html.Div(
            "Attribute classification, visual encoding choices, and layout design rationale.",
            style={
                'fontSize': '12px', 'color': C['ink_mid'],
                'fontFamily': "'DM Sans', sans-serif", 'marginBottom': '24px',
            }
        ),

        section_label("✦  ITEMS & ATTRIBUTES"),
        items_block,
        attr_table,

        section_label("✦  DATA → VISUAL CHANNEL MAPPING"),
        channel_grid,

        section_label("✦  DASHBOARD LAYOUT JUSTIFICATION"),
        layout_panel,

    ], style={
        'background': 'rgba(255,255,255,0.50)',
        'backdropFilter': 'blur(20px)',
        'WebkitBackdropFilter': 'blur(20px)',
        'border': '2px solid rgba(255,255,255,0.85)',
        'borderRadius': '24px',
        'padding': '30px 28px',
        'marginBottom': '24px',
        'boxShadow': '0 8px 40px rgba(232,24,109,0.09), inset 0 1px 0 rgba(255,255,255,0.8)',
        'position': 'relative',
        'overflow': 'hidden',
    })


# ─── App ─────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700;1,900&family=DM+Sans:wght@300;400;500;600;700;800&display=swap",
    ],
    title="DZ Products · Reddit Analysis",
)

app.index_string = '''<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  :root {
    --bg:      #FDE8EF;
    --bg2:     #FAD4E0;
    --magenta: #E8186D;
    --ink:     #2A1120;
    --ink-mid: #7A3A55;
    --ink-soft:#B07090;
    --border:  rgba(220,100,140,0.20);
  }

  body {
    background: var(--bg) !important;
    margin: 0;
    background-image:
      radial-gradient(ellipse 800px 600px at 10% 0%,   rgba(232,24,109,0.12) 0%, transparent 70%),
      radial-gradient(ellipse 600px 500px at 90% 20%,  rgba(217,70,239,0.10) 0%, transparent 70%),
      radial-gradient(ellipse 700px 500px at 50% 100%, rgba(255,140,66,0.09) 0%, transparent 70%),
      radial-gradient(ellipse 500px 400px at 80% 80%,  rgba(20,184,166,0.08) 0%, transparent 70%);
    background-attachment: fixed;
  }

  * { box-sizing: border-box; }

  h1,h2,h3,h4,h5,h6 { font-family: 'Playfair Display', serif !important; }
  p,span,div,label,input { font-family: 'DM Sans', sans-serif !important; }

  .Select-control {
    background: rgba(255,255,255,0.80) !important;
    border: 1.5px solid rgba(232,24,109,0.25) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(8px);
  }
  .Select-menu-outer {
    background: rgba(255,240,246,0.97) !important;
    border: 1.5px solid rgba(232,24,109,0.25) !important;
    border-radius: 0 0 12px 12px !important;
    z-index: 9999 !important;
    position: absolute !important;
  }
  .Select-option { color: var(--ink) !important; background: transparent !important; }
  .Select-option:hover,
  .Select-option.is-focused { background: rgba(232,24,109,0.08) !important; }
  .Select-value-label, .Select-placeholder { color: var(--ink-mid) !important; }

  .filter-bar { position: relative; z-index: 200; }

  .rc-slider-track { background: linear-gradient(90deg, #E8186D, #D946EF) !important; }
  .rc-slider-handle { border-color: #E8186D !important; background: white !important; box-shadow: 0 0 0 3px rgba(232,24,109,0.25) !important; }
  .rc-slider-rail { background: rgba(232,24,109,0.15) !important; }
  .rc-slider-mark-text { color: var(--ink-soft) !important; font-size: 10px !important; }

  .dash-checklist label { color: var(--ink) !important; }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg2); }
  ::-webkit-scrollbar-thumb { background: rgba(232,24,109,0.35); border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(232,24,109,0.55); }

  /* ── Chart cards ─────────────────────────────────────────── */
  .chart-card {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
  }
  .chart-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(232,24,109,0.14) !important;
  }
  .chart-card:hover .expand-hint {
    opacity: 1 !important;
  }

  /* ── Expand hint inside chart cards ─────────────────────── */
  .expand-hint {
    position: absolute;
    top: 10px; right: 10px;
    background: rgba(232,24,109,0.10);
    border: 1px solid rgba(232,24,109,0.28);
    color: #E8186D;
    border-radius: 20px; padding: 4px 12px;
    display: flex; align-items: center; gap: 4px;
    font-family: 'DM Sans', sans-serif;
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    z-index: 10;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.18s ease;
  }

  /* ── Modal animation ─────────────────────────────────────── */
  #modal-panel.open {
    transform: translate(-50%, -50%) scale(1) !important;
  }
  #modal-backdrop.open {
    display: block !important;
  }

  /* ── Modal scrollbar ─────────────────────────────────────── */
  #modal-panel::-webkit-scrollbar { width: 5px; }
  #modal-panel::-webkit-scrollbar-track { background: rgba(253,232,239,0.5); }
  #modal-panel::-webkit-scrollbar-thumb { background: rgba(232,24,109,0.3); border-radius: 4px; }

  .section-label {
    font-size: 10px; font-weight: 800; letter-spacing: 0.22em;
    color: var(--ink-soft); text-transform: uppercase;
    margin-bottom: 14px; display: flex; align-items: center; gap: 10px;
  }
  .section-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(232,24,109,0.25), transparent);
  }

  tbody tr:hover td { background: rgba(232,24,109,0.04) !important; }

  /* Close button hover */
  #modal-close:hover {
    background: rgba(232,24,109,0.08) !important;
    color: #E8186D !important;
  }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
<script>
// Add/remove .open class to animate modal open/close
const observer = new MutationObserver(() => {
  const panel = document.getElementById('modal-panel');
  const backdrop = document.getElementById('modal-backdrop');
  if (!panel || !backdrop) return;
  if (panel.style.display !== 'none') {
    setTimeout(() => { panel.classList.add('open'); backdrop.classList.add('open'); }, 10);
  } else {
    panel.classList.remove('open');
    backdrop.classList.remove('open');
  }
});
document.addEventListener('DOMContentLoaded', () => {
  const panel = document.getElementById('modal-panel');
  const backdrop = document.getElementById('modal-backdrop');
  if (panel) observer.observe(panel, { attributes: true, attributeFilter: ['style'] });
  // Close modal on backdrop click
  if (backdrop) {
    backdrop.addEventListener('click', () => {
      // Trigger the close button click to go through Dash callback
      const closeBtn = document.getElementById('modal-close');
      if (closeBtn) closeBtn.click();
    });
  }
  // Close on Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      const closeBtn = document.getElementById('modal-close');
      if (closeBtn) closeBtn.click();
    }
  });
});
</script>
</body>
</html>'''

# ─── Layout ──────────────────────────────────────────────────────────────────
ALL_CHART_IDS = [
    'fig-donut', 'fig-score-bar', 'fig-top-posts',
    'fig-timeline', 'fig-scatter',
    'fig-keywords', 'fig-violin',
    'fig-heatmap', 'fig-knn',
]

# Store for currently open chart id and its figure
app.layout = html.Div(
    style={'minHeight': '100vh'},
    children=[

        # Hidden stores
        dcc.Store(id='active-chart-id', data=None),
        dcc.Store(id='modal-open', data=False),

        # ── MODAL (rendered at top level, above everything) ──────────────────
        build_modal(),

        # ── HEADER ──────────────────────────────────────────────────────────
        html.Div([
            dbc.Container(fluid=True, children=[
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Div([reddit_icon(28)], style={
                                'width': '52px', 'height': '52px', 'borderRadius': '16px',
                                'background': 'rgba(255,255,255,0.80)',
                                'backdropFilter': 'blur(8px)',
                                'border': '1.5px solid rgba(255,255,255,0.95)',
                                'display': 'flex', 'alignItems': 'center',
                                'justifyContent': 'center',
                                'marginRight': '16px', 'flexShrink': '0',
                                'boxShadow': '0 4px 16px rgba(232,24,109,0.15)',
                            }),
                            html.Div([
                                html.Div([
                                    html.Span("DZ", style={
                                        'fontFamily': "'Playfair Display', serif",
                                        'fontWeight': '900', 'fontStyle': 'italic',
                                        'fontSize': '22px', 'color': C['magenta'],
                                    }),
                                    html.Span(" Products Analysis", style={
                                        'fontFamily': "'Playfair Display', serif",
                                        'fontWeight': '700', 'fontSize': '22px',
                                        'color': C['ink'],
                                    }),
                                ]),
                                html.Div([
                                    reddit_icon(12),
                                    html.Span(
                                        "Reddit consumer reviews · r/algeriarates & r/algeria · 2026",
                                        style={'fontSize': '11px', 'color': C['ink_soft'], 'marginLeft': '5px'},
                                    ),
                                ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '4px'}),
                            ]),
                        ], style={'display': 'flex', 'alignItems': 'center'}),
                    ], md=7),
                    dbc.Col([
                        html.Div([
                            html.Div("430 posts · 4 categories · 2 subreddits", style={
                                'color': C['ink_soft'], 'fontSize': '12px', 'marginBottom': '5px',
                            }),
                            html.Div([
                                html.Span("✦  made by  ", style={'color': C['ink_soft'], 'fontSize': '11px'}),
                                html.Span("Chadli Sarah Nada", style={
                                    'color': C['magenta'], 'fontSize': '12px',
                                    'fontWeight': '700', 'fontFamily': "'Playfair Display', serif",
                                    'fontStyle': 'italic',
                                }),
                                html.Span("  &  ", style={'color': C['ink_soft'], 'fontSize': '11px'}),
                                html.Span("Tachache Aya", style={
                                    'color': C['magenta'], 'fontSize': '12px',
                                    'fontWeight': '700', 'fontFamily': "'Playfair Display', serif",
                                    'fontStyle': 'italic',
                                }),
                            ]),
                        ], style={'textAlign': 'right'}),
                    ], md=5),
                ], align='center'),
            ]),
        ], style={
            'background': 'rgba(255,255,255,0.72)',
            'backdropFilter': 'blur(20px)',
            'WebkitBackdropFilter': 'blur(20px)',
            'borderBottom': '1.5px solid rgba(255,255,255,0.9)',
            'padding': '16px 24px',
            'marginBottom': '24px',
            'boxShadow': '0 4px 24px rgba(232,24,109,0.08)',
            'position': 'sticky', 'top': '0', 'zIndex': '100',
        }),

        dbc.Container(fluid=True, style={'maxWidth': '1380px', 'padding': '0 20px'}, children=[

            dataset_intro(),
            data_documentation(),

            # ── FILTER BAR ──────────────────────────────────────────────────
            html.Div([
                html.Div("✦  FILTER & EXPLORE", className="section-label"),
                dbc.Row([
                    dbc.Col([
                        html.Div("Categories", style={
                            'fontSize': '10px', 'fontWeight': '700',
                            'color': C['ink_soft'], 'textTransform': 'uppercase',
                            'letterSpacing': '0.14em', 'marginBottom': '10px',
                        }),
                        dcc.Checklist(
                            id='cat-filter',
                            options=[{
                                'label': html.Span([
                                    html.Span("●", style={
                                        'color': CAT_COLORS.get(c,'#ccc'),
                                        'marginRight': '5px', 'fontSize': '13px',
                                    }),
                                    html.Span(c.capitalize(), style={
                                        'fontWeight': '600', 'color': C['ink'], 'fontSize': '12px',
                                    }),
                                ]),
                                'value': c,
                            } for c in CATEGORIES],
                            value=CATEGORIES, inline=True,
                            labelStyle={
                                'marginRight': '22px', 'cursor': 'pointer',
                                'display': 'inline-flex', 'alignItems': 'center',
                            },
                        ),
                    ], md=6),
                    dbc.Col([
                        html.Div("Min Score", style={
                            'fontSize': '10px', 'fontWeight': '700',
                            'color': C['ink_soft'], 'textTransform': 'uppercase',
                            'letterSpacing': '0.14em', 'marginBottom': '10px',
                        }),
                        dcc.Slider(
                            id='score-slider', min=0, max=100, step=5, value=0,
                            marks={0: '0', 25: '25', 50: '50', 75: '75', 100: '100+'},
                            tooltip={'placement': 'bottom', 'always_visible': False},
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Div("Subreddit", style={
                            'fontSize': '10px', 'fontWeight': '700',
                            'color': C['ink_soft'], 'textTransform': 'uppercase',
                            'letterSpacing': '0.14em', 'marginBottom': '10px',
                        }),
                        dcc.Dropdown(
                            id='sub-filter',
                            options=[{'label': f'r/{s}', 'value': s} for s in df['subreddit'].unique()],
                            value=None, placeholder='All subreddits',
                        ),
                    ], md=2),
                ], align='start'),
            ], className="filter-bar", style={
                'background': 'rgba(255,255,255,0.68)',
                'backdropFilter': 'blur(14px)',
                'WebkitBackdropFilter': 'blur(14px)',
                'border': '1.5px solid rgba(255,255,255,0.92)',
                'borderRadius': '20px',
                'padding': '20px 24px',
                'marginBottom': '20px',
                'boxShadow': '0 4px 20px rgba(232,24,109,0.07)',
            }),

            html.Div("✦  OVERVIEW METRICS", className="section-label"),
            html.Div(id='kpi-row'),

            html.Div("✦  DISTRIBUTION & SCORES", className="section-label"),
            dbc.Row([
                dbc.Col(chart_card_with_hint('fig-donut',     '340px'), md=4),
                dbc.Col(chart_card_with_hint('fig-score-bar', '340px'), md=4),
                dbc.Col(chart_card_with_hint('fig-top-posts', '340px'), md=4),
            ]),

            html.Div("✦  ACTIVITY OVER TIME", className="section-label"),
            dbc.Row([
                dbc.Col(chart_card_with_hint('fig-timeline', '320px'), md=7),
                dbc.Col(chart_card_with_hint('fig-scatter',  '320px'), md=5),
            ]),

            html.Div("✦  TEXT ANALYSIS", className="section-label"),
            dbc.Row([
                dbc.Col(chart_card_with_hint('fig-keywords', '320px'), md=6),
                dbc.Col(chart_card_with_hint('fig-violin',   '320px'), md=6),
            ]),

            html.Div("✦  PATTERNS & MACHINE LEARNING", className="section-label"),
            dbc.Row([
                dbc.Col(chart_card_with_hint('fig-heatmap', '320px'), md=7),
                dbc.Col(chart_card_with_hint('fig-knn',     '320px'), md=5),
            ]),

            # ── FOOTER ─────────────────────────────────────────────────────
            html.Div([
                html.Div("✦", style={'color': C['magenta'], 'fontSize': '18px', 'marginBottom': '8px'}),
                html.Div([
                    html.Span("DZ Products Analysis", style={
                        'fontFamily': "'Playfair Display', serif",
                        'fontStyle': 'italic', 'fontWeight': '700',
                        'color': C['ink'], 'fontSize': '14px',
                    }),
                    html.Span("  ·  Built with Dash & Plotly", style={'color': C['ink_soft'], 'fontSize': '11px'}),
                ]),
                html.Div([
                    html.Span("Chadli Sarah Nada  &  Tachache Aya", style={
                        'fontFamily': "'Playfair Display', serif",
                        'fontStyle': 'italic', 'color': C['magenta'],
                        'fontSize': '12px', 'fontWeight': '700',
                    }),
                ], style={'marginTop': '4px'}),
            ], style={'textAlign': 'center', 'padding': '28px 0 40px'}),
        ]),
    ],
)


# ─── Callbacks ────────────────────────────────────────────────────────────────
def get_filtered(cats, sub, min_score):
    cats = cats or CATEGORIES
    mask = df['category'].isin(cats) & (df['score'] >= min_score)
    if sub:
        mask = mask & (df['subreddit'] == sub)
    return df[mask]

@app.callback(
    Output('kpi-row', 'children'),
    Input('cat-filter', 'value'),
    Input('sub-filter', 'value'),
    Input('score-slider', 'value'),
)
def update_kpis(cats, sub, min_score):
    d = get_filtered(cats, sub, min_score)
    n            = len(d)
    avg_score    = round(d['score'].mean(), 1)    if n else 0
    avg_comments = round(d['comments'].mean(), 1) if n else 0
    avg_len      = int(d['text_length'].mean())   if n else 0
    return dbc.Row([
        kpi_card("Total Posts",     str(n),            "📄", C['magenta']),
        kpi_card("Avg Score",       str(avg_score),    "⬆️",  C['tangerine']),
        kpi_card("Avg Comments",    str(avg_comments), "💬", C['teal']),
        kpi_card("Avg Text Length", str(avg_len),      "📝", C['violet']),
    ], style={'marginBottom': '10px'})

@app.callback(
    Output('fig-donut',     'figure'),
    Output('fig-score-bar', 'figure'),
    Output('fig-top-posts', 'figure'),
    Output('fig-timeline',  'figure'),
    Output('fig-scatter',   'figure'),
    Output('fig-keywords',  'figure'),
    Output('fig-violin',    'figure'),
    Output('fig-heatmap',   'figure'),
    Output('fig-knn',       'figure'),
    Input('cat-filter',   'value'),
    Input('sub-filter',   'value'),
    Input('score-slider', 'value'),
)
def update_all_charts(cats, sub, min_score):
    d = get_filtered(cats, sub, min_score)
    return (
        fig_donut(d), fig_score_bar(d), fig_top_posts(d),
        fig_timeline(d), fig_scatter(d), fig_keywords(d),
        fig_violin(d), fig_heatmap(d), fig_knn(),
    )

# ─── Modal open/close callbacks ───────────────────────────────────────────────
# We listen to n_clicks on each card-div wrapper (not clickData on the graph).
# The graph inside each card has pointerEvents:none so all clicks hit the Div.
CARD_IDS = [f'card-{cid}' for cid in ALL_CHART_IDS]

@app.callback(
    Output('modal-panel',     'style'),
    Output('modal-backdrop',  'style'),
    Output('modal-content',   'children'),
    Output('active-chart-id', 'data'),
    # n_clicks on every card div + close button + backdrop
    [Input(cid, 'n_clicks') for cid in CARD_IDS] +
    [Input('modal-close', 'n_clicks'),
     Input('modal-backdrop', 'n_clicks')],
    # current figure state for each chart (to pass into the modal)
    [State(cid, 'figure') for cid in ALL_CHART_IDS],
    prevent_initial_call=True,
)
def handle_modal(*args):
    n_cards = len(CARD_IDS)
    card_clicks   = args[:n_cards]        # n_clicks for each card div
    close_clicks  = args[n_cards]         # n_clicks for close button
    backdrop_clicks = args[n_cards + 1]   # n_clicks for backdrop
    chart_figures = args[n_cards + 2:]    # figure State for each chart

    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    MODAL_OPEN_STYLE = {
        'display': 'block',
        'position': 'fixed',
        'top': '50%', 'left': '50%',
        'transform': 'translate(-50%, -50%) scale(1)',
        'width': 'min(1100px, 96vw)',
        'maxHeight': '92vh',
        'overflowY': 'auto',
        'background': 'rgba(255,240,246,0.98)',
        'backdropFilter': 'blur(24px)',
        'WebkitBackdropFilter': 'blur(24px)',
        'border': '2px solid rgba(255,255,255,0.95)',
        'borderRadius': '28px',
        'padding': '32px',
        'boxShadow': '0 32px 80px rgba(42,17,32,0.28), inset 0 1px 0 rgba(255,255,255,0.9)',
        'zIndex': '9001',
        'transition': 'transform 0.25s ease',
    }
    BACKDROP_OPEN_STYLE = {
        'display': 'block',
        'position': 'fixed', 'inset': '0',
        'background': 'rgba(42,17,32,0.60)',
        'backdropFilter': 'blur(6px)',
        'WebkitBackdropFilter': 'blur(6px)',
        'zIndex': '9000',
        'cursor': 'pointer',
    }
    MODAL_CLOSED_STYLE    = {'display': 'none'}
    BACKDROP_CLOSED_STYLE = {'display': 'none'}

    # ── Close button or backdrop clicked ────────────────────────────────────
    if triggered_id in ('modal-close', 'modal-backdrop'):
        return MODAL_CLOSED_STYLE, BACKDROP_CLOSED_STYLE, None, None

    # ── A card div was clicked ───────────────────────────────────────────────
    if triggered_id in CARD_IDS:
        card_idx  = CARD_IDS.index(triggered_id)
        n_clicks  = card_clicks[card_idx]
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        chart_id  = ALL_CHART_IDS[card_idx]   # e.g. 'fig-donut'
        figure    = chart_figures[card_idx]
        content   = build_modal_content(chart_id, figure)
        return MODAL_OPEN_STYLE, BACKDROP_OPEN_STYLE, content, chart_id

    raise dash.exceptions.PreventUpdate

if __name__ == '__main__':
    app.run(debug=True, port=8050)