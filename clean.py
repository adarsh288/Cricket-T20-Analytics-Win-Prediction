"""
Data cleaning module for cricket analytics.
Takes raw scraped CSVs from /data/raw and produces cleaned versions in /data/processed.
Handles missing values, inconsistent names, date formatting, and duplicates.
"""

import pandas as pd
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define file paths
RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'


def clean_batting_summary():
    """
    Clean batting summary data.
    
    Real-world messiness this fixes:
    - Missing values (e.g., players who didn't bat)
    - Inconsistent team names (e.g., "India" vs "India ")
    - Numeric fields stored as strings (runs, balls, SR)
    - Duplicate rows from scraping errors
    
    Why it matters for analysis:
    - Missing values need handling before aggregations (e.g., calculating average runs)
    - Consistent team names needed for joining with other tables
    - Numeric strings break calculations (can't sum "25" as text)
    - Duplicates inflate statistics artificially
    """
    logger.info("Cleaning batting summary data...")
    
    # Read raw data
    input_path = os.path.join(RAW_DIR, 'batting_summary.csv')
    if not os.path.exists(input_path):
        logger.warning(f"File not found: {input_path}")
        return None
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from raw data")
    
    # Remove duplicate rows
    # Why: Scraping might retry and insert same record twice
    df = df.drop_duplicates()
    logger.info(f"After removing duplicates: {len(df)} rows")
    
    # Strip whitespace from string columns
    # Why: "India " and "India" should be treated as same team
    string_cols = ['match', 'teamInnings', 'batsmanName', 'dismissal']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].str.strip()
    
    # Handle missing values in numeric columns
    # Why: Some players might have "not out" or empty dismissal, but runs/balls should be numbers
    numeric_cols = ['runs', 'balls', '4s', '6s', 'SR']
    for col in numeric_cols:
        if col in df.columns:
            # Convert to numeric, coerce errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill NaN with 0 for runs/balls (player didn't face ball)
            if col in ['runs', 'balls', '4s', '6s']:
                df[col] = df[col].fillna(0)
            # Leave SR as NaN if can't calculate (division by zero case)
    
    # Clean dismissal column - standardize common variations
    # Why: "not out", "not out ", "Not Out" should all be consistent
    if 'dismissal' in df.columns:
        df['dismissal'] = df['dismissal'].str.lower().str.strip()
        df['dismissal'] = df['dismissal'].replace('not out', 'not out')
    
    # Ensure batting position is integer
    # Why: Position should be 1, 2, 3... not 1.0 or string
    if 'battingPos' in df.columns:
        df['battingPos'] = pd.to_numeric(df['battingPos'], errors='coerce').fillna(0).astype(int)
    
    # Remove rows with missing essential fields
    # Why: Can't analyze a record without knowing who played or which match
    essential_cols = ['match', 'teamInnings', 'batsmanName']
    df = df.dropna(subset=essential_cols)
    
    logger.info(f"Final cleaned data: {len(df)} rows")
    
    # Save to processed directory
    output_path = os.path.join(PROCESSED_DIR, 'batting_summary_cleaned.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")
    
    return df


def clean_bowling_summary():
    """
    Clean bowling summary data.
    
    Real-world messiness this fixes:
    - Missing values (e.g., bowlers who didn't bowl)
    - Inconsistent team names
    - Numeric fields stored as strings (overs, wickets, economy)
    - Special characters in bowler names
    
    Why it matters for analysis:
    - Economy rate calculations need numeric values
    - Team consistency needed for head-to-head analysis
    - Wicket counts must be accurate for bowler rankings
    """
    logger.info("Cleaning bowling summary data...")
    
    input_path = os.path.join(RAW_DIR, 'bowling_summary.csv')
    if not os.path.exists(input_path):
        logger.warning(f"File not found: {input_path}")
        return None
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from raw data")
    
    # Remove duplicates
    df = df.drop_duplicates()
    logger.info(f"After removing duplicates: {len(df)} rows")
    
    # Strip whitespace from string columns
    string_cols = ['match', 'bowlingTeam', 'bowlerName']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].str.strip()
    
    # Handle numeric columns
    numeric_cols = ['overs', 'maiden', 'runs', 'wickets', 'economy', '0s', '4s', '6s', 'wides', 'noBalls']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill NaN with 0 for count columns
            if col != 'economy':
                df[col] = df[col].fillna(0)
    
    # Remove rows with missing essential fields
    essential_cols = ['match', 'bowlingTeam', 'bowlerName']
    df = df.dropna(subset=essential_cols)
    
    logger.info(f"Final cleaned data: {len(df)} rows")
    
    output_path = os.path.join(PROCESSED_DIR, 'bowling_summary_cleaned.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")
    
    return df


def clean_player_info():
    """
    Clean player information data.
    
    Real-world messiness this fixes:
    - Duplicate player entries (same player in multiple teams/matches)
    - Missing playing role or style information
    - Inconsistent team names
    - Empty or very short descriptions
    
    Why it matters for analysis:
    - Player-level analysis needs unique player records
    - Playing role helps categorize players (batsman, bowler, all-rounder)
    - Team consistency needed for joining with match data
    """
    logger.info("Cleaning player info data...")
    
    input_path = os.path.join(RAW_DIR, 'player_info.csv')
    if not os.path.exists(input_path):
        logger.warning(f"File not found: {input_path}")
        return None
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from raw data")
    
    # Remove duplicates based on player name (keep first occurrence)
    # Why: Same player scraped from multiple matches should be one record
    if 'name' in df.columns:
        df = df.drop_duplicates(subset=['name'], keep='first')
    logger.info(f"After removing duplicate players: {len(df)} rows")
    
    # Strip whitespace
    string_cols = ['name', 'team', 'battingStyle', 'bowlingStyle', 'playingRole', 'description']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Standardize empty strings to NaN for better handling
    # Why: Empty string and NaN should be treated the same way
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].replace('', pd.NA)
    
    # Remove rows without player name
    if 'name' in df.columns:
        df = df.dropna(subset=['name'])
    
    logger.info(f"Final cleaned data: {len(df)} rows")
    
    output_path = os.path.join(PROCESSED_DIR, 'player_info_cleaned.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")
    
    return df


def clean_match_results():
    """
    Clean match results data.
    
    Real-world messiness this fixes:
    - Inconsistent date formats (e.g., "Jan 15, 2024" vs "15-01-2024")
    - Missing margin information (for tied/abandoned matches)
    - Inconsistent team names
    - Ground name variations
    
    Why it matters for analysis:
    - Date consistency needed for time-series analysis
    - Margin needed for understanding match competitiveness
    - Team names needed for joining with player performance data
    - Ground names needed for venue-based analysis
    """
    logger.info("Cleaning match results data...")
    
    input_path = os.path.join(RAW_DIR, 'match_results.csv')
    if not os.path.exists(input_path):
        logger.warning(f"File not found: {input_path}")
        return None
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from raw data")
    
    # Remove duplicates
    df = df.drop_duplicates()
    logger.info(f"After removing duplicates: {len(df)} rows")
    
    # Strip whitespace from string columns
    string_cols = ['team1', 'team2', 'winner', 'margin', 'ground', 'scorecard']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Standardize date format
    # Why: ESPN might use different date formats; we want consistent YYYY-MM-DD
    if 'matchDate' in df.columns:
        df['matchDate'] = pd.to_datetime(df['matchDate'], errors='coerce')
        # Format as YYYY-MM-DD string for consistency
        df['matchDate'] = df['matchDate'].dt.strftime('%Y-%m-%d')
    
    # Handle missing margin (tied matches, no result)
    # Why: These are valid match outcomes, just without a win margin
    if 'margin' in df.columns:
        df['margin'] = df['margin'].replace('nan', pd.NA)
    
    # Remove rows with missing essential fields
    essential_cols = ['team1', 'team2', 'matchDate']
    df = df.dropna(subset=essential_cols)
    
    logger.info(f"Final cleaned data: {len(df)} rows")
    
    output_path = os.path.join(PROCESSED_DIR, 'match_results_cleaned.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")
    
    return df


def clean_all_data():
    """
    Run all cleaning functions.
    
    Why this function: Convenience to clean all datasets at once with one command.
    Useful when running the full pipeline from scrape to analysis.
    """
    logger.info("Starting full data cleaning pipeline...")
    
    # Create processed directory if it doesn't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Clean each dataset
    batting_df = clean_batting_summary()
    bowling_df = clean_bowling_summary()
    player_df = clean_player_info()
    match_df = clean_match_results()
    
    logger.info("Data cleaning pipeline complete!")
    
    return {
        'batting': batting_df,
        'bowling': bowling_df,
        'player': player_df,
        'match': match_df
    }


if __name__ == '__main__':
    # Run all cleaning when script is executed directly
    clean_all_data()
