# Spotify Hit Predictor & Recommender

A Streamlit app that analyzes Spotify track data and helps predict whether a song is likely to become a hit, estimates popularity, and recommends similar songs based on audio features.

## Features

- Dataset overview with summary statistics and visual analysis
- Linear regression model to predict popularity score from audio features
- Logistic regression model to classify songs as hits vs. non-hits
- K-nearest neighbors (KNN) recommender for similar song suggestions
- Interactive sliders and filters for custom predictions

## Project Structure

- `app.py` — main Streamlit application
- `data/spotify_songs.csv` — Spotify music dataset used by the models
- `requirements.txt` — Python dependencies

## Tech Stack

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn

## Setup

1. Open a terminal in the project folder.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## Notes

- The app expects the dataset file at `data/spotify_songs.csv`.
- Some models use popularity thresholds and audio features such as danceability, energy, valence, tempo, loudness, acousticness, and speechiness.
- The app is designed for exploratory music analytics and simple predictive modeling.

## License

This project is provided for educational and demo purposes.
