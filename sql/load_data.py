"""
Load processed CSV data into SQLite database.
Handles foreign key relationships by mapping names to IDs.
"""

import sqlite3
import pandas as pd
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths - use script-relative paths to ensure consistency regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data/processed')
DB_PATH = os.path.join(PROJECT_ROOT, 'cricket_t20.db')
SCHEMA_PATH = os.path.join(PROJECT_ROOT, 'sql', 'schema.sql')


def create_database():
    """
    Create SQLite database and run schema.sql.
    
    Why: Need to set up tables before loading data.
    Schema.sql defines the structure and relationships.
    """
    logger.info("Creating database schema...")
    
    # Read schema SQL
    with open(SCHEMA_PATH, 'r') as f:
        schema_sql = f.read()
    
    # Create connection and execute schema
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(schema_sql)
    conn.commit()
    conn.close()
    
    logger.info("Database schema created successfully")


def load_teams_and_matches():
    """
    Load match_results.csv to populate teams and matches tables.
    
    Why this order:
    - Teams must exist first because matches reference team IDs
    - Matches need team1_id, team2_id, winner_id as foreign keys
    
    Process:
    1. Extract unique team names from match_results
    2. Insert teams, get their auto-generated IDs
    3. Map team names to IDs
    4. Insert matches with team IDs instead of names
    """
    logger.info("Loading teams and matches...")
    
    # Read match results
    match_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'match_results_cleaned.csv'))
    logger.info(f"Loaded {len(match_df)} match records")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Extract unique team names
    # Why: Need to insert each team only once
    team_names = set(match_df['team1'].tolist() + match_df['team2'].tolist())
    if 'winner' in match_df.columns:
        team_names.update(match_df['winner'].dropna().tolist())
    
    logger.info(f"Found {len(team_names)} unique teams")
    
    # Insert teams and build name->id mapping
    team_name_to_id = {}
    for team_name in team_names:
        if pd.notna(team_name) and team_name.strip():
            cursor.execute(
                "INSERT OR IGNORE INTO teams (team_name) VALUES (?)",
                (team_name.strip(),)
            )
            conn.commit()
    
    # Get the IDs for all teams
    cursor.execute("SELECT team_id, team_name FROM teams")
    for team_id, team_name in cursor.fetchall():
        team_name_to_id[team_name] = team_id
    
    logger.info(f"Inserted teams with IDs: {team_name_to_id}")
    
    # Insert matches
    # Why: Map team names to IDs for foreign key constraints
    for _, row in match_df.iterrows():
        team1_name = row['team1']
        team2_name = row['team2']
        winner_name = row.get('winner', None)
        margin = row.get('margin', None)
        ground = row.get('ground', None)
        match_date = row.get('matchDate', None)
        match_id = row.get('match_id', None)
        
        # Get team IDs
        team1_id = team_name_to_id.get(team1_name)
        team2_id = team_name_to_id.get(team2_name)
        winner_id = team_name_to_id.get(winner_name) if pd.notna(winner_name) else None
        
        # Insert match
        cursor.execute(
            """INSERT OR REPLACE INTO matches 
               (match_id, team1_id, team2_id, winner_id, margin, ground, match_date)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (match_id, team1_id, team2_id, winner_id, margin, ground, match_date)
        )
    
    conn.commit()
    conn.close()
    logger.info("Teams and matches loaded successfully")


def load_players():
    """
    Load player_info.csv to populate players table.
    
    Why: Players need team_id foreign key, so teams must exist first.
    Also need to handle potential duplicates (same player name).
    
    Process:
    1. Read player info
    2. Map team names to team IDs (from teams table)
    3. Insert players with team_id
    """
    logger.info("Loading players...")
    
    # Read player info
    player_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'player_info_cleaned.csv'))
    logger.info(f"Loaded {len(player_df)} player records")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get team name to ID mapping
    cursor.execute("SELECT team_id, team_name FROM teams")
    team_name_to_id = {team_name: team_id for team_id, team_name in cursor.fetchall()}
    
    # Insert players
    for _, row in player_df.iterrows():
        name = row['name']
        team_name = row.get('team', None)
        image = row.get('image', None)
        batting_style = row.get('battingStyle', None)
        bowling_style = row.get('bowlingStyle', None)
        playing_role = row.get('playingRole', None)
        description = row.get('description', None)
        
        # Get team ID
        team_id = team_name_to_id.get(team_name) if pd.notna(team_name) else None
        
        # Insert player (UNIQUE constraint on name, team_id handles duplicates)
        cursor.execute(
            """INSERT OR IGNORE INTO players 
               (name, team_id, image, batting_style, bowling_style, playing_role, description)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, team_id, image, batting_style, bowling_style, playing_role, description)
        )
    
    conn.commit()
    conn.close()
    logger.info("Players loaded successfully")


def load_batting():
    """
    Load batting_summary.csv to populate batting table.
    
    Why: Batting records need match_id, team_id, and player_id foreign keys.
    Must map names to IDs from existing tables.
    
    Process:
    1. Read batting summary
    2. Map match_id, team names, and player names to IDs
    3. Insert batting records with foreign keys
    """
    logger.info("Loading batting summary...")
    
    # Read batting summary
    batting_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'batting_summary_cleaned.csv'))
    logger.info(f"Loaded {len(batting_df)} batting records")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get mappings
    cursor.execute("SELECT team_id, team_name FROM teams")
    team_name_to_id = {team_name: team_id for team_id, team_name in cursor.fetchall()}
    
    cursor.execute("SELECT player_id, name FROM players")
    player_name_to_id = {name: player_id for player_id, name in cursor.fetchall()}
    
    # Insert batting records
    for _, row in batting_df.iterrows():
        match_id = row.get('match_id', None)
        team_innings_name = row.get('teamInnings', None)
        batsman_name = row.get('batsmanName', None)
        batting_pos = row.get('battingPos', None)
        runs = row.get('runs', None)
        balls = row.get('balls', None)
        fours = row.get('4s', None)
        sixes = row.get('6s', None)
        strike_rate = row.get('SR', None)
        dismissal = row.get('dismissal', None)
        
        # Get foreign key IDs
        team_innings_id = team_name_to_id.get(team_innings_name) if pd.notna(team_innings_name) else None
        batsman_id = player_name_to_id.get(batsman_name) if pd.notna(batsman_name) else None
        
        # Insert batting record
        cursor.execute(
            """INSERT INTO batting 
               (match_id, team_innings_id, batsman_id, batting_pos, runs, balls, fours, sixes, strike_rate, dismissal)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (match_id, team_innings_id, batsman_id, batting_pos, runs, balls, fours, sixes, strike_rate, dismissal)
        )
    
    conn.commit()
    conn.close()
    logger.info("Batting summary loaded successfully")


def load_bowling():
    """
    Load bowling_summary.csv to populate bowling table.
    
    Why: Bowling records need match_id, team_id, and player_id foreign keys.
    Similar to batting but with bowling-specific columns.
    
    Process:
    1. Read bowling summary
    2. Map match_id, team names, and player names to IDs
    3. Insert bowling records with foreign keys
    """
    logger.info("Loading bowling summary...")
    
    # Read bowling summary
    bowling_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'bowling_summary_cleaned.csv'))
    logger.info(f"Loaded {len(bowling_df)} bowling records")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get mappings
    cursor.execute("SELECT team_id, team_name FROM teams")
    team_name_to_id = {team_name: team_id for team_id, team_name in cursor.fetchall()}
    
    cursor.execute("SELECT player_id, name FROM players")
    player_name_to_id = {name: player_id for player_id, name in cursor.fetchall()}
    
    # Insert bowling records
    for _, row in bowling_df.iterrows():
        match_id = row.get('match_id', None)
        bowling_team_name = row.get('bowlingTeam', None)
        bowler_name = row.get('bowlerName', None)
        overs = row.get('overs', None)
        maiden = row.get('maiden', None)
        runs = row.get('runs', None)
        wickets = row.get('wickets', None)
        economy = row.get('economy', None)
        zeros = row.get('0s', None)
        fours = row.get('4s', None)
        sixes = row.get('6s', None)
        wides = row.get('wides', None)
        no_balls = row.get('noBalls', None)
        
        # Get foreign key IDs
        bowling_team_id = team_name_to_id.get(bowling_team_name) if pd.notna(bowling_team_name) else None
        bowler_id = player_name_to_id.get(bowler_name) if pd.notna(bowler_name) else None
        
        # Insert bowling record
        cursor.execute(
            """INSERT INTO bowling 
               (match_id, bowling_team_id, bowler_id, overs, maiden, runs, wickets, economy, zeros, fours, sixes, wides, no_balls)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (match_id, bowling_team_id, bowler_id, overs, maiden, runs, wickets, economy, zeros, fours, sixes, wides, no_balls)
        )
    
    conn.commit()
    conn.close()
    logger.info("Bowling summary loaded successfully")


def load_all_data():
    """
    Run the complete data loading pipeline.
    
    Why this order is critical:
    1. Create schema (tables must exist)
    2. Load teams + matches (teams needed for FKs)
    3. Load players (needs team IDs)
    4. Load batting (needs match, team, player IDs)
    5. Load bowling (needs match, team, player IDs)
    """
    logger.info("Starting data loading pipeline...")
    
    # Check if processed files exist
    required_files = [
        'match_results_cleaned.csv',
        'player_info_cleaned.csv',
        'batting_summary_cleaned.csv',
        'bowling_summary_cleaned.csv'
    ]
    
    for file in required_files:
        file_path = os.path.join(PROCESSED_DIR, file)
        if not os.path.exists(file_path):
            logger.error(f"Required file not found: {file_path}")
            logger.error("Please run clean.py first to generate processed files")
            return
    
    # Create database schema
    create_database()
    
    # Load data in correct order
    load_teams_and_matches()
    load_players()
    load_batting()
    load_bowling()
    
    logger.info("Data loading pipeline complete!")
    logger.info(f"Database created at: {DB_PATH}")


if __name__ == '__main__':
    load_all_data()
