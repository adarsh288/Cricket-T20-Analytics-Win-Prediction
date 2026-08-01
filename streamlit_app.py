"""
Streamlit app for Cricket T20 Win Prediction.

Overall structure:
- Tab 1: Dashboard - Shows analytical charts from SQL queries (venue performance, head-to-head, etc.)
- Tab 2: Predict a Match - Interactive prediction with team/venue dropdowns and SHAP explanations
- Tab 3: Insights - Model explanation, why XGBoost was chosen, known limitations

Key design decisions:
- Reuse feature computation logic from features.py to avoid duplication
- Simple clean styling, not flashy
- Honest framing about model accuracy (~60% CV) and directional nature of predictions
- SHAP explanations for interpretability

Dependencies: streamlit, pandas, sqlite3, plotly, shap, sklearn, xgboost
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import pickle
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import xgboost as xgb
from datetime import datetime

# Import feature computation logic from features.py
# Why: Reuse existing functions to avoid duplicating logic and ensure consistency
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from features import (
    calculate_rolling_win_pct,
    calculate_head_to_head,
    calculate_venue_advantage,
    is_neutral_venue,
    prepare_feature_matrix
)

# Paths - use script-relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
DB_PATH = os.path.join(PROJECT_ROOT, 'cricket_t20.db')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'saved_model.pkl')

# Page configuration
st.set_page_config(
    page_title="Cricket T20 Win Prediction",
    page_icon="",
    layout="wide"
)

# Custom CSS for background gradient
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 40%, #0d1b2a 100%);
}
</style>
""", unsafe_allow_html=True)

st.title("Cricket T20 Win Prediction")
st.markdown("---")


@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects."""
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


@st.cache_data
def get_teams():
    """Get list of teams from the database."""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT team_name FROM teams ORDER BY team_name"
    teams = pd.read_sql_query(query, conn)
    conn.close()
    return teams['team_name'].tolist()


@st.cache_data
def get_venues():
    """Get list of venues from the database."""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT DISTINCT ground FROM matches WHERE ground IS NOT NULL ORDER BY ground"
    venues = pd.read_sql_query(query, conn)
    conn.close()
    return venues['ground'].tolist()


@st.cache_data
def get_match_data_for_features():
    """Load match data for feature computation."""
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        m.match_id,
        t1.team_name AS team1,
        t2.team_name AS team2,
        CASE 
            WHEN m.winner_id = m.team1_id THEN 1
            WHEN m.winner_id = m.team2_id THEN 0
            ELSE NULL
        END AS team1_won,
        m.ground,
        m.match_date
    FROM matches m
    JOIN teams t1 ON m.team1_id = t1.team_id
    JOIN teams t2 ON m.team2_id = t2.team_id
    WHERE m.winner_id IS NOT NULL
    ORDER BY m.match_date
    """
    df = pd.read_sql_query(query, conn)
    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    df = df.dropna(subset=['match_date'])
    conn.close()
    return df


def compute_features_for_match(team1, team2, ground, df):
    """
    Compute the 5 features for a hypothetical matchup using historical data.
    Reuses the same logic as features.py to ensure consistency.
    """
    # Create a temporary row for the hypothetical match
    today = datetime.now()
    temp_row = pd.DataFrame([{
        'match_id': 999999,
        'team1': team1,
        'team2': team2,
        'team1_won': None,  # Unknown for prediction
        'ground': ground,
        'match_date': today
    }])
    
    # Append to historical data
    df_with_temp = pd.concat([df, temp_row], ignore_index=True)
    df_with_temp = df_with_temp.sort_values('match_date').reset_index(drop=True)
    
    # Compute rolling win percentages
    df_with_temp['team1_rolling_win_pct'] = df_with_temp.groupby('team1').apply(
        lambda g: calculate_rolling_win_pct(g, 'team1')
    ).reset_index(level=0, drop=True)
    
    df_with_temp['team2_rolling_win_pct'] = df_with_temp.groupby('team2').apply(
        lambda g: calculate_rolling_win_pct(g, 'team2')
    ).reset_index(level=0, drop=True)
    
    # Compute head-to-head
    df_with_temp['head_to_head_win_pct'] = df_with_temp.apply(
        lambda row: calculate_head_to_head(row, df_with_temp), axis=1
    )
    
    # Compute venue advantage
    df_with_temp['venue_advantage'] = df_with_temp.apply(
        lambda row: calculate_venue_advantage(row, df_with_temp), axis=1
    )
    
    # Compute neutral venue flag
    df_with_temp['is_neutral_venue'] = df_with_temp.apply(
        lambda row: is_neutral_venue(row), axis=1
    )
    
    # Get features for the hypothetical match (last row)
    features = df_with_temp.iloc[-1][[
        'team1_rolling_win_pct',
        'team2_rolling_win_pct',
        'head_to_head_win_pct',
        'venue_advantage',
        'is_neutral_venue'
    ]].values
    
    return features


# Load model and data
model_data = load_model()
model = model_data['model']
model_name = model_data['model_name']
scaler = model_data['scaler']
feature_names = model_data['feature_names']
lr_cv_mean = model_data.get('lr_cv_mean')
lr_cv_std = model_data.get('lr_cv_std')
xgb_cv_mean = model_data.get('xgb_cv_mean')
xgb_cv_std = model_data.get('xgb_cv_std')

teams = get_teams()
venues = get_venues()
match_df = get_match_data_for_features()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Dashboard", "Predict a Match", "Insights"])

# TAB 1: Dashboard
with tab1:
    st.header("Dashboard")
    st.markdown("Analytical insights from the cricket T20 data.")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Chart 1: Top run scorers by venue
    st.subheader("Top Run Scorers by Venue")
    batting_query = """
    SELECT 
        b.batsman_id,
        p.name AS batsman_name,
        m.ground,
        SUM(b.runs) AS total_runs
    FROM batting b
    JOIN players p ON b.batsman_id = p.player_id
    JOIN matches m ON b.match_id = m.match_id
    GROUP BY b.batsman_id, p.name, m.ground
    ORDER BY m.ground, total_runs DESC
    """
    batting_by_venue = pd.read_sql_query(batting_query, conn)
    
    if not batting_by_venue.empty:
        top_scorers = batting_by_venue.groupby('ground').first().reset_index()
        fig1 = px.bar(
            top_scorers.head(10),
            x='total_runs',
            y='ground',
            color='batsman_name',
            title='Top Run Scorer by Venue',
            labels={'total_runs': 'Total Runs', 'ground': 'Venue', 'batsman_name': 'Batsman'},
            orientation='h'
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No batting data available for venue analysis.")
    
    # Chart 2: Head-to-head records between top teams
    st.subheader("Head-to-Head Records")
    h2h_query = """
    SELECT 
        t1.team_name AS team1,
        t2.team_name AS team2,
        COUNT(*) AS matches_played,
        SUM(CASE WHEN m.winner_id = m.team1_id THEN 1 ELSE 0 END) AS team1_wins
    FROM matches m
    JOIN teams t1 ON m.team1_id = t1.team_id
    JOIN teams t2 ON m.team2_id = t2.team_id
    GROUP BY t1.team_name, t2.team_name
    HAVING matches_played >= 2
    ORDER BY matches_played DESC
    LIMIT 10
    """
    h2h_data = pd.read_sql_query(h2h_query, conn)
    
    if not h2h_data.empty:
        h2h_data['team2_wins'] = h2h_data['matches_played'] - h2h_data['team1_wins']
        fig2 = go.Figure(data=[
            go.Bar(name='Team 1 Wins', x=h2h_data['team1'] + ' vs ' + h2h_data['team2'], y=h2h_data['team1_wins']),
            go.Bar(name='Team 2 Wins', x=h2h_data['team1'] + ' vs ' + h2h_data['team2'], y=h2h_data['team2_wins'])
        ])
        fig2.update_layout(barmode='group', title='Head-to-Head Records (Top Matchups)')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Insufficient match data for head-to-head analysis.")
    
    # Chart 3: Win percentage by venue
    st.subheader("Team Performance by Venue")
    venue_query = """
    SELECT 
        t.team_name,
        m.ground,
        COUNT(*) AS matches_played,
        SUM(CASE WHEN 
            (m.team1_id = t.team_id AND m.winner_id = m.team1_id) OR
            (m.team2_id = t.team_id AND m.winner_id = m.team2_id)
            THEN 1 ELSE 0 END) AS wins
    FROM matches m
    CROSS JOIN teams t
    WHERE m.team1_id = t.team_id OR m.team2_id = t.team_id
    GROUP BY t.team_name, m.ground
    HAVING matches_played >= 2
    ORDER BY matches_played DESC
    LIMIT 15
    """
    venue_data = pd.read_sql_query(venue_query, conn)
    
    if not venue_data.empty:
        venue_data['win_pct'] = (venue_data['wins'] / venue_data['matches_played'] * 100).round(1)
        fig3 = px.scatter(
            venue_data,
            x='matches_played',
            y='win_pct',
            color='team_name',
            size='matches_played',
            hover_data=['ground'],
            title='Team Win Percentage by Venue',
            labels={'matches_played': 'Matches Played', 'win_pct': 'Win %'}
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Insufficient venue data for performance analysis.")
    
    conn.close()


# TAB 2: Predict a Match
with tab2:
    st.header("Predict a Match")
    st.markdown("Select teams and venue to predict the win probability.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        team1 = st.selectbox("Team 1", teams, index=0)
    
    with col2:
        team2 = st.selectbox("Team 2", teams, index=min(1, len(teams)-1))
    
    with col3:
        ground = st.selectbox("Venue", venues, index=0)
    
    if st.button("Predict Winner", type="primary"):
        if team1 == team2:
            st.error("Please select different teams for the matchup.")
        else:
            with st.spinner("Computing features and making prediction..."):
                # Compute features
                features = compute_features_for_match(team1, team2, ground, match_df)
                
                # Convert to DataFrame with proper column names to match training format
                # Why: XGBoost was trained on DataFrame with named columns, needs same structure at prediction time
                feature_cols = [
                    'team1_rolling_win_pct',
                    'team2_rolling_win_pct',
                    'head_to_head_win_pct',
                    'venue_advantage',
                    'is_neutral_venue'
                ]
                features_df = pd.DataFrame([features], columns=feature_cols)
                
                # Apply scaling if using Logistic Regression
                if scaler is not None:
                    features_scaled = scaler.transform(features_df)
                    proba = model.predict_proba(features_scaled)[0, 1]
                else:
                    proba = model.predict_proba(features_df)[0, 1]
                
                # Display prediction
                team1_win_pct = proba * 100
                team2_win_pct = 100 - team1_win_pct
                
                st.subheader("Prediction")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(f"{team1} Win Probability", f"{team1_win_pct:.1f}%")
                with col_b:
                    st.metric(f"{team2} Win Probability", f"{team2_win_pct:.1f}%")
                
                # Progress bar visualization
                st.progress(team1_win_pct / 100)
                st.caption(f"{team1} (left) vs {team2} (right)")
                
                # Feature contribution explanation
                st.subheader("Feature Contributions")
                
                # Generate feature contributions using XGBoost's native prediction contributions
                # Why: Avoids SHAP's persistent base_score parsing error with XGBoost models
                with st.spinner("Computing feature contributions..."):
                    # Prepare features for contribution computation (apply scaling if needed)
                    if scaler is not None:
                        features_for_contrib = scaler.transform(features_df)[0]
                    else:
                        features_for_contrib = features_df.values[0]
                    
                    # Debug: Print feature order and values
                    print(f"DEBUG feature_names: {feature_names}")
                    print(f"DEBUG features_for_contrib: {features_for_contrib}")
                    print(f"DEBUG raw features_df:\n{features_df}")
                    print(f"DEBUG team1={team1}, team2={team2}, ground={ground}")
                    
                    if model_name == "XGBoost":
                        # Use XGBoost's native prediction contributions
                        dmatrix = xgb.DMatrix(features_for_contrib.reshape(1, -1), feature_names=feature_names)
                        contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
                        base_value = contribs[0][-1]
                        feature_contributions = contribs[0][:-1]
                    else:
                        # For Logistic Regression, use coefficients as contributions
                        feature_contributions = model.coef_[0] * features_for_contrib
                        base_value = model.intercept_[0]
                    
                    # Build manual bar chart of feature contributions
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Sort by absolute value for waterfall-like reading
                    sorted_indices = np.argsort(np.abs(feature_contributions))
                    sorted_contribs = feature_contributions[sorted_indices]
                    sorted_features = [feature_names[i] for i in sorted_indices]
                    
                    # Color by sign (positive = green, negative = red)
                    colors = ['green' if c > 0 else 'red' for c in sorted_contribs]
                    
                    # Horizontal bar chart
                    y_pos = np.arange(len(sorted_features))
                    ax.barh(y_pos, sorted_contribs, color=colors, alpha=0.7)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(sorted_features)
                    ax.set_xlabel('Contribution to Prediction')
                    ax.set_title('Feature Contributions to Win Probability')
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                    
                    # Add base value annotation
                    ax.text(0.02, 0.98, f'Base Value: {base_value:.3f}', 
                           transform=ax.transAxes, fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # Plain-language explanation
                st.subheader("What's Driving This Prediction?")
                
                # Find most influential features based on feature contributions
                contrib_abs = np.abs(feature_contributions)
                top_indices = np.argsort(contrib_abs)[-2:][::-1]  # Top 2 features
                
                explanation = "The prediction is primarily driven by "
                if len(top_indices) >= 2:
                    explanation += f"{feature_names[top_indices[0]]} and {feature_names[top_indices[1]].lower()}. "
                elif len(top_indices) == 1:
                    explanation += f"{feature_names[top_indices[0]]}. "
                else:
                    explanation += "the combination of all features. "
                
                explanation += "Recent form and head-to-head records are typically the biggest factors in T20 predictions."
                st.info(explanation)
                
                # Model accuracy disclaimer
                st.caption(f"Model cross-validated accuracy: ~{lr_cv_mean*100:.0f}% if Logistic Regression, "
                          f"~{xgb_cv_mean*100:.0f}% if XGBoost. Predictions are directional indicators, "
                          f"not certainties. Small dataset (45 matches) limits reliability.")


# TAB 3: Insights
with tab3:
    st.header("Insights")
    
    st.subheader("What the Model Uses to Predict")
    st.markdown("""
    The model uses 5 key features engineered from historical match data:
    
    - **Rolling Win Percentage**: Each team's win rate over their last 5 matches (captures current form)
    - **Head-to-Head Record**: Team 1's historical win percentage against Team 2 (rivalry factor)
    - **Venue Advantage**: Whether Team 1 has historically won more at this venue (home/away effect)
    - **Neutral Venue Flag**: Binary flag indicating if the venue is neutral (removes home advantage)
    
    Why these features matter:
    - Recent form is a strong predictor in T20 cricket - teams on winning streaks tend to continue
    - Some teams consistently dominate others regardless of current form
    - Home teams often have crowd support and pitch familiarity
    - Neutral venues level the playing field by removing home advantage
    """)
    
    st.subheader("Why XGBoost Was Chosen")
    if xgb_cv_mean is not None and lr_cv_mean is not None:
        st.markdown(f"""
        **Cross-Validation Performance:**
        - Logistic Regression: {lr_cv_mean:.2f} ± {lr_cv_std:.2f}
        - XGBoost: {xgb_cv_mean:.2f} ± {xgb_cv_std:.2f}
        
        XGBoost was selected because:
        - **Better mean accuracy**: {xgb_cv_mean:.2f} vs {lr_cv_mean:.2f} for Logistic Regression
        - **More consistent**: Lower standard deviation ({xgb_cv_std:.2f} vs {lr_cv_std:.2f}) across folds
        - **Captures non-linear relationships**: Tree-based model can learn complex feature interactions
        
        Logistic Regression remains a useful baseline for interpretability, but XGBoost's superior performance
        justifies its additional complexity for this prediction task.
        """)
    else:
        st.info("Cross-validation metrics not available in saved model.")
    
    st.subheader("Known Limitations")
    st.markdown("""
    **Dataset Size:**
    - Single tournament with only 45 matches limits statistical reliability
    - Small sample size means predictions should be read directionally, not as precise forecasts
    - More data across multiple tournaments would improve model robustness
    
    **Missing Features:**
    - Toss data (winner and bat/field decision) is not available in the current schema
    - Toss is a significant factor in T20 cricket - chasing teams often have an advantage
    - Player injuries, team composition changes, and weather conditions are not captured
    
    **Model Interpretation:**
    - Cross-validated accuracy of ~60% indicates the model captures signal but has substantial uncertainty
    - Use predictions as one input among many, not as the sole decision factor
    - Model performance may vary across different teams and venues not well-represented in training data
    """)
    
    st.subheader("How to Use This Tool")
    st.markdown("""
    1. **Select teams and venue** in the Predict a Match tab
    2. **Review the prediction** - the win probability shows the model's confidence
    3. **Check feature contributions** - understand which factors are driving the prediction
    4. **Consider the limitations** - remember this is a directional indicator, not a guarantee
    5. **Use alongside other information** - team news, expert analysis, and current conditions
    
    This tool is designed to augment cricket analysis with data-driven insights, not replace human judgment.
    """)


st.markdown("---")
st.caption("Built with Streamlit | Model: XGBoost/Logistic Regression | Data: ESPN Cricinfo")
