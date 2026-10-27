import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score

# Page Configuration
st.set_page_config(page_title="Spotify Hit Predictor & Recommender", layout="wide")

st.title("🎵 Spotify Music Hit Predictor & Recommender")

# Load Data from 'data' folder
@st.cache_data
def load_data():
    return pd.read_csv("data/spotify_songs.csv")

try:
    df = load_data()
except Exception as e:
    st.error("Error loading `data/spotify_songs.csv`. Make sure the CSV file is placed inside the `data` folder.")
    st.stop()

# Audio features list
feature_cols = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'acousticness', 'speechiness']

# Sidebar Navigation
option = st.sidebar.selectbox("Choose Module", ["Dataset Overview & EDA", "1. Linear Regression (Popularity Score)", "2. Logistic Regression (Hit Predictor)", "3. KNN Song Recommender"])

# -----------------------------------------------------------------------------
# MODULE 0: EDA
# -----------------------------------------------------------------------------
if option == "Dataset Overview & EDA":
    st.header("🔍 Dataset Overview & Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(10))
    with col2:
        st.subheader("Summary Statistics")
        st.write(df.describe())
        
    st.markdown("---")
    st.subheader("Visualizations")
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Popularity Distribution**")
        fig, ax = plt.subplots()
        sns.histplot(df['popularity'], kde=True, ax=ax, color='purple')
        st.pyplot(fig)
        
    with c2:
        st.write("**Feature Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[feature_cols + ['popularity']].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# MODULE 1: LINEAR REGRESSION
# -----------------------------------------------------------------------------
elif option == "1. Linear Regression (Popularity Score)":
    st.header("📈 Linear Regression: Predict Popularity Score (0 - 100)")
    
    X = df[feature_cols]
    y = df['popularity']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    col2.metric("R² Score", f"{r2:.2f}")
    
    st.subheader("🎛️ Test Custom Audio Features")
    st.write("Adjust features below to predict track popularity score:")
    
    inputs = {}
    cols = st.columns(3)
    for i, col_name in enumerate(feature_cols):
        min_val = float(df[col_name].min())
        max_val = float(df[col_name].max())
        mean_val = float(df[col_name].mean())
        inputs[col_name] = cols[i % 3].slider(col_name, min_value=min_val, max_value=max_val, value=mean_val)
        
    input_df = pd.DataFrame([inputs])
    predicted_score = model.predict(input_df)[0]
    
    st.success(f"**Predicted Popularity Score:** `{np.clip(predicted_score, 0, 100):.2f} / 100`")

# -----------------------------------------------------------------------------
# MODULE 2: LOGISTIC REGRESSION
# -----------------------------------------------------------------------------
elif option == "2. Logistic Regression (Hit Predictor)":
    st.header("🎯 Logistic Regression: Is it a HIT?")
    
    threshold = st.slider("Define Popularity Threshold for a 'HIT'", min_value=50, max_value=85, value=70)
    
    df['is_hit'] = (df['popularity'] >= threshold).astype(int)
    
    st.write(f"Class Distribution (Hit = 1, Not Hit = 0):")
    st.write(df['is_hit'].value_counts())
    
    X = df[feature_cols]
    y = df['is_hit']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression()
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    st.metric("Model Accuracy", f"{acc * 100:.2f}%")
    
    st.subheader("🧪 Predict New Song Hit Probability")
    
    inputs = {}
    cols = st.columns(3)
    for i, col_name in enumerate(feature_cols):
        min_val = float(df[col_name].min())
        max_val = float(df[col_name].max())
        mean_val = float(df[col_name].mean())
        inputs[col_name] = cols[i % 3].slider(f"{col_name} ", min_value=min_val, max_value=max_val, value=mean_val)
        
    input_df = pd.DataFrame([inputs])
    input_scaled = scaler.transform(input_df)
    
    hit_prob = clf.predict_proba(input_scaled)[0][1]
    is_hit_pred = clf.predict(input_scaled)[0]
    
    st.write("---")
    st.write(f"### Hit Probability: **{hit_prob * 100:.2f}%**")
    if is_hit_pred == 1:
        st.balloons()
        st.success("🔥 Prediction: **This song is going to be a HIT!**")
    else:
        st.warning("❄️ Prediction: **This song might not cross the Hit threshold.**")

# -----------------------------------------------------------------------------
# MODULE 3: KNN RECOMMENDER
# -----------------------------------------------------------------------------
elif option == "3. KNN Song Recommender":
    st.header("🎧 KNN Audio Feature Similarity Recommender")
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    
    k = st.slider("Number of Recommendations (K)", min_value=3, max_value=10, value=5)
    
    knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    knn.fit(scaled_features)
    
    selected_song = st.selectbox("Select a song to get recommendations:", df['track_name'].values)
    
    if st.button("Get Recommendations"):
        song_idx = df[df['track_name'] == selected_song].index[0]
        song_vector = scaled_features[song_idx].reshape(1, -1)
        
        distances, indices = knn.kneighbors(song_vector)
        
        st.subheader(f"Recommendations similar to **{selected_song}**:")
        
        rec_indices = indices[0][1:] # exclude the query song itself
        recommendations = df.iloc[rec_indices][['track_name', 'artist_name', 'popularity'] + feature_cols]
        
        st.dataframe(recommendations)