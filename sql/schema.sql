-- SQLite schema for Cricket T20 Analytics
-- Normalized structure: matches, teams, players, batting, bowling

-- Why this normalization: 
-- - teams table: Avoids repeating team names across multiple tables
-- - players table: Single source of truth for player info (style, role, description)
-- - matches table: Central table that ties everything together via match_id
-- - batting/bowling: Performance tables with FKs to matches, teams, and players

-- Teams table
-- Why: Normalize team names so "India" is consistent across all records
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Players table
-- Why: Store player profile info once instead of repeating in every match record
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    team_id INTEGER,
    image TEXT,
    batting_style TEXT,
    bowling_style TEXT,
    playing_role TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    UNIQUE(name, team_id)
);

-- Matches table
-- Why: Central table that captures match-level info and ties everything together
CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    team1_id INTEGER,
    team2_id INTEGER,
    winner_id INTEGER,
    margin TEXT,
    ground TEXT,
    match_date TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (team1_id) REFERENCES teams(team_id),
    FOREIGN KEY (team2_id) REFERENCES teams(team_id),
    FOREIGN KEY (winner_id) REFERENCES teams(team_id)
);

-- Batting summary table
-- Why: Store batting performance with FKs to match, team, and player for easy joins
CREATE TABLE IF NOT EXISTS batting (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT,
    team_innings_id INTEGER,
    batsman_id INTEGER,
    batting_pos INTEGER,
    runs INTEGER,
    balls INTEGER,
    fours INTEGER,
    sixes INTEGER,
    strike_rate TEXT,
    dismissal TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (team_innings_id) REFERENCES teams(team_id),
    FOREIGN KEY (batsman_id) REFERENCES players(player_id)
);

-- Bowling summary table
-- Why: Store bowling performance with FKs to match, team, and player for easy joins
CREATE TABLE IF NOT EXISTS bowling (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT,
    bowling_team_id INTEGER,
    bowler_id INTEGER,
    overs REAL,
    maiden INTEGER,
    runs INTEGER,
    wickets INTEGER,
    economy REAL,
    zeros INTEGER,
    fours INTEGER,
    sixes INTEGER,
    wides INTEGER,
    no_balls INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (bowling_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (bowler_id) REFERENCES players(player_id)
);

-- Create indexes for common query patterns
-- Why: Speed up queries that filter by match, player, or team
CREATE INDEX IF NOT EXISTS idx_batting_match ON batting(match_id);
CREATE INDEX IF NOT EXISTS idx_batting_player ON batting(batsman_id);
CREATE INDEX IF NOT EXISTS idx_bowling_match ON bowling(match_id);
CREATE INDEX IF NOT EXISTS idx_bowling_player ON bowling(bowler_id);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_ground ON matches(ground);
