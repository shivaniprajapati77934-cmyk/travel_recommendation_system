# 🌍 Wanderly: Travel Recommendation System

A data science project (IBM 7th Semester) that recommends travel destinations from a dataset of 35 global destinations.

## Features
- **Preference-based recommender**: filter by season, budget, continent, crowds and safety, then rank by a weighted 0-100 match score.
- **Similarity-based recommender**: find destinations similar to one you like (cosine similarity).
- **Data exploration**: interactive charts, a country map and a correlation heatmap.
- **Streamlit web app** and a **Jupyter notebook** with the full analysis.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `travel_recommendation_ds_project.ipynb` | Notebook: cleaning, EDA, recommenders |
| `global_travel_destinations.csv` | Dataset |
| `requirements.txt` | Python dependencies |

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Live demo
https://travel-recommendation-system-do2za9ytc4zkofbscygwg8.streamlit.app/


## Tech
Python, pandas, scikit-learn, seaborn, matplotlib, plotly, Streamlit.
