"""
Feature engineering for cricket match win prediction.
Extracts features from SQLite database for machine learning models.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use script-relative path to ensure database is found regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, 'cricket_t20.db')


def get_matches_with_features():
    """
    Load matches data and engineer features for prediction.
    
    Features engineered:
    - team1_rolling_win_pct: Team 1's win rate in recent matches (form indicator)
    - team2_rolling_win_pct: Team 2's win rate in recent matches (form indicator)
    - head_to_head_win_pct: Team 1's historical win % against Team 2 (rivalry factor)
    - venue_advantage: Whether Team 1 has historically won more at this venue (home/away effect)
    - is_neutral_venue: Binary flag for neutral venues (affects home advantage)
    
    Why these features matter:
    - Rolling win %: Captures current form - teams on winning streaks tend to keep winning
    - Head-to-head: Some teams consistently dominate others regardless of form
    - Venue advantage: Home teams often have crowd support and pitch familiarity
    - Neutral venue: Removes home advantage, levels the playing field
    
    KNOWN LIMITATION: Toss result feature not included because current schema lacks toss data.
    Toss winner and decision (bat/field) would be valuable features but require schema update.
    """
    logger.info("Loading matches and engineering features...")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Load base match data from normalized schema (matches + teams tables)
    # Why: Schema uses foreign keys - matches.team1_id joins to teams.team_name
    # Use ID comparison to determine if team1 won (winner_id == team1_id)
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
    logger.info(f"Loaded {len(df)} matches")
    
    # Convert date string to datetime for sorting
    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    df = df.dropna(subset=['match_date'])
    
    # Feature 1: Rolling win percentage for each team
    # Why: Recent form is a strong predictor - teams playing well tend to continue
    df['team1_rolling_win_pct'] = df.groupby('team1').apply(
        lambda x: calculate_rolling_win_pct(x, 'team1')
    ).reset_index(level=0, drop=True)
    
    df['team2_rolling_win_pct'] = df.groupby('team2').apply(
        lambda x: calculate_rolling_win_pct(x, 'team2')
    ).reset_index(level=0, drop=True)
    
    # Feature 2: Head-to-head win percentage
    # Why: Historical dominance - some teams have psychological edge over others
    df['head_to_head_win_pct'] = df.apply(
        lambda row: calculate_head_to_head(row, df), axis=1
    )
    
    # Feature 3: Venue advantage for Team 1
    # Why: Home teams often perform better due to crowd support and pitch familiarity
    df['venue_advantage'] = df.apply(
        lambda row: calculate_venue_advantage(row, df), axis=1
    )
    
    # Feature 4: Neutral venue flag
    # Why: Neutral venues remove home advantage, making matches more unpredictable
    # Simple heuristic: if neither team is from the venue's country, it's neutral
    df['is_neutral_venue'] = df.apply(
        lambda row: is_neutral_venue(row), axis=1
    )
    
    conn.close()
    
    # Drop rows with missing target or features
    df = df.dropna(subset=['team1_won'])
    df = df.dropna(subset=['team1_rolling_win_pct', 'team2_rolling_win_pct', 
                           'head_to_head_win_pct', 'venue_advantage'])
    
    logger.info(f"Final dataset: {len(df)} matches with features")
    
    return df


def calculate_rolling_win_pct(group, team_col):
    """
    Calculate rolling 5-match win percentage for a team.
    
    Why 5 matches: Captures recent form without being too noisy.
    Too few matches = unstable, too many = outdated form.
    
    Args:
        group: DataFrame group for one team
        team_col: 'team1' or 'team2' column name
        
    Returns:
        Series of rolling win percentages
    """
    # Determine if this team won each match
    if team_col == 'team1':
        wins = group['team1_won']
    else:
        wins = 1 - group['team1_won']  # Team 2 wins when team1 doesn't
    
    # Calculate rolling average over last 5 matches
    # Why shift by 1: To prevent data leakage - only include matches BEFORE current match
    # Without shift, the rolling window includes the current match's result
    rolling_pct = wins.shift(1).rolling(window=5, min_periods=1).mean()
    
    return rolling_pct


def calculate_head_to_head(row, full_df):
    """
    Calculate Team 1's historical win % against Team 2 before this match.
    
    Why: Some teams consistently dominate others (e.g., Australia vs Bangladesh).
    This captures psychological and tactical advantages.
    
    Args:
        row: Current match row
        full_df: Full matches dataframe
        
    Returns:
        Win percentage (0-1)
    """
    team1 = row['team1']
    team2 = row['team2']
    current_date = row['match_date']
    
    # Get previous matches between these two teams
    previous_matches = full_df[
        ((full_df['team1'] == team1) & (full_df['team2'] == team2)) |
        ((full_df['team1'] == team2) & (full_df['team2'] == team1))
    ]
    previous_matches = previous_matches[previous_matches['match_date'] < current_date]
    
    if len(previous_matches) == 0:
        return 0.5  # No history, assume neutral (50%)
    
    # Count Team 1 wins
    team1_wins = 0
    for _, match in previous_matches.iterrows():
        if match['team1'] == team1 and match['team1_won'] == 1:
            team1_wins += 1
        elif match['team2'] == team1 and match['team1_won'] == 0:
            team1_wins += 1
    
    return team1_wins / len(previous_matches)


def calculate_venue_advantage(row, full_df):
    """
    Calculate Team 1's historical win % at this venue.
    
    Why: Teams often perform better at familiar venues (home advantage).
    Pitch conditions, crowd support, and familiarity all play a role.
    
    Args:
        row: Current match row
        full_df: Full matches dataframe
        
    Returns:
        Win percentage at venue (0-1)
    """
    team1 = row['team1']
    ground = row['ground']
    current_date = row['match_date']
    
    # Get previous Team 1 matches at this venue
    previous_matches = full_df[
        (full_df['team1'] == team1) & 
        (full_df['ground'] == ground) &
        (full_df['match_date'] < current_date)
    ]
    
    if len(previous_matches) == 0:
        return 0.5  # No history at venue, assume neutral
    
    # Calculate win percentage
    wins = previous_matches['team1_won'].sum()
    return wins / len(previous_matches)


def is_neutral_venue(row):
    """
    Determine if venue is neutral (neither team's home ground).
    
    Why: Neutral venues remove home advantage, making predictions harder.
    Simple heuristic based on team names and venue names.
    
    KNOWN LIMITATION: This is a rough heuristic. A proper implementation
    would need a venue-to-country mapping table.
    
    Args:
        row: Current match row
        
    Returns:
        1 if neutral, 0 if not neutral
    """
    team1 = row['team1'].lower()
    team2 = row['team2'].lower()
    ground = row['ground'].lower() if pd.notna(row['ground']) else ''
    
    # Simple heuristic: if venue name doesn't contain either team's country name, it's neutral
    # This is imperfect but works for major cricket venues
    if team1 in ground or team2 in ground:
        return 0  # Not neutral (at least one team's home)
    else:
        return 1  # Neutral


def prepare_feature_matrix(df):
    """
    Prepare feature matrix X and target vector y for modeling.
    
    Why: Separate features from target, handle categorical variables,
    and ensure proper data types for sklearn/xgboost.
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
    """
    # Select feature columns
    feature_cols = [
        'team1_rolling_win_pct',
        'team2_rolling_win_pct',
        'head_to_head_win_pct',
        'venue_advantage',
        'is_neutral_venue'
    ]
    
    X = df[feature_cols].copy()
    y = df['team1_won'].copy()
    
    # Ensure numeric types
    X = X.astype(float)
    y = y.astype(int)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Feature names: {feature_cols}")
    
    return X, y, feature_cols


if __name__ == '__main__':
    # Test feature engineering
    df = get_matches_with_features()
    X, y, feature_names = prepare_feature_matrix(df)
    
    print("\nFeature matrix sample:")
    print(X.head())
    print("\nTarget distribution:")
    print(y.value_counts())
