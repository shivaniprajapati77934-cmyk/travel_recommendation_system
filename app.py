"""
Wanderly - Travel Recommendation System
Run with:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path(__file__).parent / "global_travel_destinations.csv"

# ----------------------------------------------------------------------------
# Recommendation logic (same code as the notebook)
# ----------------------------------------------------------------------------

CROWD_MAP = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
TEXT_COLS = ["destination", "country", "continent", "best_season", "crowd_level"]
NUM_COLS = ["temperature", "rainfall", "aqi", "safety", "budget", "popularity"]
COLUMN_RENAMES = {
    "avg_temperature_c": "temperature",
    "avg_rainfall_mm": "rainfall",
    "avg_aqi": "aqi",
    "safety_rating_5": "safety",
    "estimated_daily_budget_usd": "budget",
    "popularity_score_100": "popularity",
}


def clean_data(raw):
    """Standardise column names, fix types, drop duplicates and incomplete rows."""
    df = raw.copy()
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns=COLUMN_RENAMES)
    for col in TEXT_COLS:
        df[col] = df[col].astype(str).str.strip()
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates()
    df = df.dropna(subset=TEXT_COLS + NUM_COLS)
    return df.reset_index(drop=True)

def add_features(df):
    """Add helper columns used by the recommender and the charts."""
    df = df.copy()
    df["crowd_score"] = df["crowd_level"].map(CROWD_MAP)
    df["budget_tier"] = pd.cut(
        df["budget"], bins=[0, 80, 150, 10_000],
        labels=["Budget", "Mid-range", "Premium"],
    ).astype(str)
    return df


def build_similarity(df):
    """Cosine similarity between destinations using numbers + continent + season."""
    numeric = df[NUM_COLS + ["crowd_score"]]
    scaled = StandardScaler().fit_transform(numeric)
    categories = pd.get_dummies(df[["continent", "best_season"]]).astype(float).to_numpy() * 0.5
    return cosine_similarity(np.hstack([scaled, categories]))

def minmax(s):
    return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0


def apply_filters(df, seasons=None, max_budget=None, continents=None, crowd=None, min_safety=0.0):
    out = df
    if seasons:
        out = out[out["best_season"].isin(seasons)]
    if continents:
        out = out[out["continent"].isin(continents)]
    if max_budget is not None:
        out = out[out["budget"] <= max_budget]
    if crowd and crowd != "Any":
        out = out[out["crowd_level"] == crowd]
    if min_safety:
        out = out[out["safety"] >= min_safety]
    return out


def similar_destinations(df, sim, name, top_n=5, **filters):
    """Destinations most similar to `name`, optionally filtered."""
    if name not in df["destination"].values:
        raise ValueError(f"'{name}' not found. Check the spelling.")
    idx = df.index[df["destination"] == name][0]
    out = df.copy()
    out["similarity"] = ((sim[idx] + 1) / 2 * 100).round(1)   # cosine (-1..1) -> 0..100 %
    out = out.drop(index=idx)
    out = apply_filters(out, **filters)
    return out.sort_values("similarity", ascending=False).head(top_n)

DEFAULT_WEIGHTS = {"safety": 3, "budget": 3, "air_quality": 2, "popularity": 2, "temperature": 3}
HIGHLIGHT_LABELS = {
    "safety": "High safety", "budget": "Budget friendly", "air_quality": "Clean air",
    "popularity": "Very popular", "temperature": "Matches your temperature",
}


def recommend_by_preferences(df, seasons=None, max_budget=None, continents=None, crowd=None,
                             min_safety=0.0, ideal_temp=None, weights=None, top_n=5):
    """Filter by hard constraints, then rank by a weighted match score (0-100)."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    parts = {
        "safety": minmax(df["safety"]),
        "budget": 1 - minmax(df["budget"]),          # cheaper = better
        "air_quality": 1 - minmax(df["aqi"]),        # lower AQI = better
        "popularity": minmax(df["popularity"]),
    }
    if ideal_temp is not None:
        gap = (df["temperature"] - ideal_temp).abs()
        parts["temperature"] = (1 - gap / 15).clip(lower=0)
    used = {k: w[k] for k in parts if w.get(k, 0) > 0} or {k: 1 for k in parts}
    total = sum(used.values())
    score = sum(parts[k] * wt for k, wt in used.items()) / total * 100

    out = apply_filters(df, seasons, max_budget, continents, crowd, min_safety).copy()
    out["match_score"] = score.loc[out.index].round(1)
    out["highlights"] = [
        " - ".join(HIGHLIGHT_LABELS[k] for k in used if parts[k].loc[i] >= 0.75)
        for i in out.index
    ]
    return out.sort_values("match_score", ascending=False).head(top_n)


# ----------------------------------------------------------------------------
# Streamlit interface
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Wanderly | Travel Recommender", page_icon="🌍", layout="wide")

st.markdown("""
<style>
.hero {background: linear-gradient(120deg,#0f766e,#2563eb); padding: 28px 32px; border-radius: 16px; color: white; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 2.1rem; color: white;}
.hero p {margin: 6px 0 0 0; opacity: .92; font-size: 1.05rem;}
.card {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 16px 18px; margin-bottom: 12px; background: rgba(128,128,128,.07);}
.card-top {display:flex; justify-content:space-between; align-items:center; gap:12px;}
.card-title {font-size: 1.2rem; font-weight: 700;}
.card-sub {opacity:.7; font-size:.92rem;}
.badge {background:#0f766e; color:white; padding:5px 12px; border-radius:999px; font-weight:700; white-space:nowrap;}
.chip {display:inline-block; border:1px solid rgba(128,128,128,.35); border-radius:999px; padding:3px 10px; margin:8px 6px 0 0; font-size:.85rem;}
.why {margin-top:10px; font-size:.9rem; color:#0f766e; font-weight:600;}
.bar {height:6px; border-radius:6px; background:rgba(128,128,128,.25); margin-top:12px; overflow:hidden;}
.bar > div {height:100%; background: linear-gradient(90deg,#0f766e,#2563eb);}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load():
    data = add_features(clean_data(pd.read_csv(CSV_PATH)))
    return data, build_similarity(data)


def render_card(rank, row, badge_text, pct):
    chips = "".join(
        f'<span class="chip">{t}</span>'
        for t in [f"📅 {row['best_season']}", f"🌡️ {row['temperature']}°C", f"💵 ${row['budget']}/day",
                  f"🛡️ {row['safety']}/5", f"👥 {row['crowd_level']} crowds", f"🌧️ {row['rainfall']} mm"]
    )
    why = f'<div class="why">✨ {row["highlights"]}</div>' if row.get("highlights") else ""
    st.markdown(
        f'<div class="card"><div class="card-top"><div>'
        f'<div class="card-title">#{rank} {row["destination"]}</div>'
        f'<div class="card-sub">{row["country"]} · {row["continent"]} · {row["budget_tier"]}</div></div>'
        f'<div class="badge">{badge_text}</div></div>{chips}{why}'
        f'<div class="bar"><div style="width:{max(0, min(100, pct))}%"></div></div></div>',
        unsafe_allow_html=True,
    )


def results_table(df, extra):
    cols = ["destination", "country", "continent", "best_season", "temperature", "budget", "safety", "crowd_level"]
    shown = df[[extra] + cols].rename(columns={"temperature": "temp °C", "budget": "budget $/day"})
    st.dataframe(shown, hide_index=True, width="stretch")
    st.download_button("⬇️ Download results (CSV)", shown.to_csv(index=False), "recommendations.csv", "text/csv")


df, sim = load()
seasons = sorted(df["best_season"].unique())
continents = sorted(df["continent"].unique())

st.markdown(
    '<div class="hero"><h1>🌍 Wanderly</h1>'
    "<p>Tell us how you like to travel and we'll rank the best destinations for you.</p></div>",
    unsafe_allow_html=True,
)

tab_find, tab_similar, tab_explore, tab_random = st.tabs(
    ["🎯 Find my destination", "🔁 Similar places", "📊 Explore the data", "🎲 Surprise me"]
)

# ---- Tab 1: preference-based recommendations -------------------------------
with tab_find:
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Your preferences")
        sel_seasons = st.multiselect("📅 Travel season", seasons, placeholder="Any season")
        sel_conts = st.multiselect("🗺️ Continent", continents, placeholder="Anywhere")
        max_budget = st.slider("💵 Max daily budget (USD)", int(df.budget.min()), int(df.budget.max()), int(df.budget.max()), step=5)
        crowd = st.selectbox("👥 Crowd level", ["Any"] + sorted(df["crowd_level"].unique()))
        min_safety = st.slider("🛡️ Minimum safety rating", 0.0, 5.0, 3.5, step=0.1)
        use_temp = st.checkbox("🌡️ I have a preferred temperature")
        ideal_temp = st.slider("Ideal average temperature (°C)", 10, 32, 22) if use_temp else None
        with st.expander("⚖️ What matters most? (1 = low, 5 = high)"):
            w_safety = st.slider("Safety", 0, 5, DEFAULT_WEIGHTS["safety"])
            w_budget = st.slider("Affordability", 0, 5, DEFAULT_WEIGHTS["budget"])
            w_air = st.slider("Clean air", 0, 5, DEFAULT_WEIGHTS["air_quality"])
            w_pop = st.slider("Popularity", 0, 5, DEFAULT_WEIGHTS["popularity"])
            w_temp = st.slider("Temperature match", 0, 5, DEFAULT_WEIGHTS["temperature"], disabled=not use_temp)
        top_n = st.slider("How many results?", 3, 10, 5)
    with right:
        results = recommend_by_preferences(
            df, seasons=sel_seasons, max_budget=max_budget, continents=sel_conts, crowd=crowd,
            min_safety=min_safety, ideal_temp=ideal_temp, top_n=top_n,
            weights={"safety": w_safety, "budget": w_budget, "air_quality": w_air,
                     "popularity": w_pop, "temperature": w_temp},
        )
        st.subheader("Top matches")
        if results.empty:
            st.warning("No destinations match these filters. Try a higher budget, lower safety minimum or more seasons.")
        else:
            for rank, (_, row) in enumerate(results.iterrows(), start=1):
                render_card(rank, row, f"{row['match_score']:.0f}% match", row["match_score"])
            with st.expander("View as table"):
                results_table(results, "match_score")

# ---- Tab 2: similar destinations -------------------------------------------
with tab_similar:
    c1, c2, c3 = st.columns([2, 1, 1])
    base = c1.selectbox("✈️ I liked…", sorted(df["destination"]), index=sorted(df["destination"]).index("Paris"))
    n_sim = c2.slider("Results", 3, 10, 5, key="n_sim")
    same_cont = c3.checkbox("Same continent only")
    sim_filters = {"continents": [df.loc[df.destination == base, "continent"].iloc[0]]} if same_cont else {}
    similar = similar_destinations(df, sim, base, top_n=n_sim, **sim_filters)
    st.caption(f"Destinations most like **{base}** based on climate, cost, safety, crowds, popularity, continent and season.")
    for rank, (_, row) in enumerate(similar.iterrows(), start=1):
        render_card(rank, row, f"{row['similarity']:.0f}% similar", row["similarity"])

# ---- Tab 3: explore the data -----------------------------------------------
with tab_explore:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Destinations", len(df))
    m2.metric("Countries", df["country"].nunique())
    m3.metric("Avg daily budget", f"${df['budget'].mean():.0f}")
    m4.metric("Avg safety", f"{df['safety'].mean():.2f} / 5")

    a, b = st.columns(2)
    fig = px.scatter(df, x="budget", y="safety", size="popularity", color="continent", hover_name="destination",
                     title="Budget vs safety (bubble size = popularity)", labels={"budget": "Daily budget (USD)", "safety": "Safety (1-5)"})
    a.plotly_chart(fig, width="stretch")
    fig = px.box(df, x="continent", y="budget", color="continent", points="all", hover_name="destination",
                 title="Daily budget by continent")
    fig.update_layout(showlegend=False)
    b.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    season_counts = df["best_season"].value_counts().reset_index()
    fig = px.bar(season_counts, x="best_season", y="count", color="best_season", title="Destinations by best season")
    fig.update_layout(showlegend=False, xaxis_title="Best season", yaxis_title="Destinations")
    a.plotly_chart(fig, width="stretch")
    fig = px.scatter(df, x="temperature", y="rainfall", color="best_season", hover_name="destination",
                     title="Temperature vs rainfall", labels={"temperature": "Avg temperature (°C)", "rainfall": "Avg rainfall (mm)"})
    b.plotly_chart(fig, width="stretch")

    country_budget = df.groupby("country", as_index=False)["budget"].mean()
    fig = px.choropleth(country_budget, locations="country", locationmode="country names", color="budget",
                        color_continuous_scale="Teal", title="Average daily budget by country")
    st.plotly_chart(fig, width="stretch")

    corr = df[NUM_COLS + ["crowd_score"]].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation between features")
    st.plotly_chart(fig, width="stretch")

    with st.expander("Browse the full dataset"):
        st.dataframe(df.drop(columns=["crowd_score"]), hide_index=True, width="stretch")

# ---- Tab 4: random destination ---------------------------------------------
with tab_random:
    st.subheader("Feeling adventurous?")
    if st.button("🎲 Pick a random destination", type="primary"):
        pick = df.sample(1).iloc[0].copy()
        pick["highlights"] = ""
        render_card("🎉", pick, f"{pick['popularity']} / 100 popular", pick["popularity"])