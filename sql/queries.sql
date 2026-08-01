-- Analytical queries for Cricket T20 Analytics
-- Each query answers a specific business question

-- ============================================
-- Query 1: Top run scorers by venue
-- Business Question: Which players score the most runs at each ground?
-- Why this matters: Helps identify venue specialists and inform team selection
-- for specific grounds (e.g., a player who performs well at spin-friendly pitches)
-- ============================================

WITH venue_runs AS (
    -- Join batting with matches to get ground information
    -- Sum runs by player and venue
    SELECT 
        m.ground,
        p.name AS player_name,
        t.team_name,
        SUM(b.runs) AS total_runs,
        COUNT(*) AS innings_played
    FROM batting b
    JOIN matches m ON b.match_id = m.match_id
    JOIN players p ON b.batsman_id = p.player_id
    JOIN teams t ON b.team_innings_id = t.team_id
    GROUP BY m.ground, p.name, t.team_name
),
ranked_players AS (
    -- Rank players within each venue by total runs
    SELECT 
        ground,
        player_name,
        team_name,
        total_runs,
        innings_played,
        ROW_NUMBER() OVER (PARTITION BY ground ORDER BY total_runs DESC) AS rank
    FROM venue_runs
)
-- Get top 5 scorers for each venue
SELECT 
    ground,
    player_name,
    team_name,
    total_runs,
    innings_played
FROM ranked_players
WHERE rank <= 5
ORDER BY ground, total_runs DESC;


-- ============================================
-- Query 2: Team win % by toss decision
-- Business Question: Does winning the toss actually help teams win matches?
-- Why this matters: Tests the common cricket wisdom that "toss wins matches"
-- and can inform captain's decision if they win the toss
-- 
-- KNOWN LIMITATION: Current schema does not include toss decision data.
-- This query is a template showing what would be needed if toss data were available.
-- To implement, add 'toss_winner' and 'toss_decision' columns to matches table.
-- ============================================

-- Template query (requires toss data in matches table):
/*
WITH toss_outcomes AS (
    SELECT 
        t1.team_name AS toss_winner,
        m.toss_decision,  -- 'bat' or 'field'
        CASE 
            WHEN m.winner_id = m.team1_id THEN t1.team_name
            WHEN m.winner_id = m.team2_id THEN t2.team_name
            ELSE NULL
        END AS match_winner,
        CASE 
            WHEN m.winner_id = m.team1_id AND m.toss_winner = m.team1_id THEN 'won_toss_won_match'
            WHEN m.winner_id = m.team2_id AND m.toss_winner = m.team2_id THEN 'won_toss_won_match'
            WHEN m.winner_id = m.team1_id AND m.toss_winner = m.team2_id THEN 'lost_toss_won_match'
            WHEN m.winner_id = m.team2_id AND m.toss_winner = m.team1_id THEN 'lost_toss_won_match'
            ELSE 'no_result'
        END AS outcome
    FROM matches m
    JOIN teams t1 ON m.team1_id = t1.team_id
    JOIN teams t2 ON m.team2_id = t2.team_id
)
SELECT 
    toss_winner,
    toss_decision,
    outcome,
    COUNT(*) AS matches,
    ROUND(100.0 * SUM(CASE WHEN outcome = 'won_toss_won_match' THEN 1 ELSE 0 END) / COUNT(*), 2) AS win_percentage_when_won_toss
FROM toss_outcomes
GROUP BY toss_winner, toss_decision, outcome
ORDER BY toss_winner, toss_decision;
*/

-- Alternative: Simple team win percentage (without toss data)
-- Business Question: What is each team's overall win percentage?
WITH team_matches AS (
    SELECT 
        t1.team_name,
        COUNT(*) AS total_matches
    FROM matches m
    JOIN teams t1 ON m.team1_id = t1.team_id
    GROUP BY t1.team_name
),
team_wins AS (
    SELECT 
        t.team_name,
        COUNT(*) AS wins
    FROM matches m
    JOIN teams t ON m.winner_id = t.team_id
    GROUP BY t.team_name
)
SELECT 
    tm.team_name,
    tm.total_matches,
    COALESCE(tw.wins, 0) AS wins,
    ROUND(100.0 * COALESCE(tw.wins, 0) / tm.total_matches, 2) AS win_percentage
FROM team_matches tm
LEFT JOIN team_wins tw ON tm.team_name = tw.team_name
ORDER BY win_percentage DESC;


-- ============================================
-- Query 3: Powerplay vs death-over performance by team
-- Business Question: How do teams perform in the first 6 overs (powerplay) 
-- compared to the last 5 overs (death overs)?
-- Why this matters: Identifies teams with strong/weak phases, helping with
-- bowling changes and batting strategy
--
-- KNOWN LIMITATION: Current schema has only summary stats (total runs, wickets),
-- not ball-by-ball data needed for phase-based analysis.
-- This query is a template showing what would be needed with ball-by-ball data.
-- To implement, add a ball_by_ball table with over_number and ball_number.
-- ============================================

-- Template query (requires ball-by-ball data):
/*
WITH phase_performance AS (
    SELECT 
        t.team_name,
        CASE 
            WHEN bb.over_number <= 6 THEN 'powerplay'
            WHEN bb.over_number >= 16 THEN 'death'
            ELSE 'middle'
        END AS phase,
        SUM(bb.runs_scored) AS runs,
        SUM(CASE WHEN bb.is_wicket = 1 THEN 1 ELSE 0 END) AS wickets,
        COUNT(*) AS balls
    FROM ball_by_ball bb
    JOIN matches m ON bb.match_id = m.match_id
    JOIN teams t ON bb.batting_team_id = t.team_id
    GROUP BY t.team_name, phase
)
SELECT 
    team_name,
    phase,
    runs,
    wickets,
    balls,
    ROUND(runs * 6.0 / balls, 2) AS run_rate,
    ROUND(wickets * 6.0 / balls, 2) AS wicket_rate
FROM phase_performance
WHERE phase IN ('powerplay', 'death')
ORDER BY team_name, phase;
*/

-- Alternative: Team bowling economy comparison (using available summary data)
-- Business Question: Which teams have the best bowling economy overall?
WITH team_bowling AS (
    SELECT 
        t.team_name,
        SUM(b.runs) AS total_runs_conceded,
        SUM(b.overs) AS total_overs,
        SUM(b.wickets) AS total_wickets,
        COUNT(*) AS bowling_spells
    FROM bowling b
    JOIN teams t ON b.bowling_team_id = t.team_id
    GROUP BY t.team_name
)
SELECT 
    team_name,
    total_runs_conceded,
    total_overs,
    total_wickets,
    bowling_spells,
    ROUND(total_runs_conceded / NULLIF(total_overs, 0), 2) AS economy,
    ROUND(total_wickets * 6.0 / NULLIF(total_overs, 0), 2) AS strike_rate_balls
FROM team_bowling
ORDER BY economy ASC;


-- ============================================
-- Query 4: Player form trend (rolling average across last N matches)
-- Business Question: How has a player's batting performance trended over their recent matches?
-- Why this matters: Helps identify players in good form vs slumps, useful for team selection
-- and predicting future performance
-- ============================================

-- This query shows a player's runs in their last 10 matches with a running average
-- Adjust the number in WHERE clause to change the window size

WITH player_match_runs AS (
    -- Get runs scored by player in each match, ordered by date
    SELECT 
        p.name AS player_name,
        t.team_name,
        m.match_date,
        m.match_id,
        b.runs,
        ROW_NUMBER() OVER (PARTITION BY p.name ORDER BY m.match_date) AS match_number
    FROM batting b
    JOIN matches m ON b.match_id = m.match_id
    JOIN players p ON b.batsman_id = p.player_id
    JOIN teams t ON b.team_innings_id = t.team_id
    WHERE p.name = 'Virat Kohli'  -- Replace with any player name
),
rolling_avg AS (
    -- Calculate rolling average of runs over last 5 matches
    SELECT 
        player_name,
        team_name,
        match_date,
        match_id,
        runs,
        AVG(runs) OVER (
            PARTITION BY player_name 
            ORDER BY match_date 
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) AS rolling_5_match_avg
    FROM player_match_runs
)
SELECT 
    player_name,
    team_name,
    match_date,
    runs,
    ROUND(rolling_5_match_avg, 2) AS rolling_5_match_avg
FROM rolling_avg
ORDER BY match_date;


-- ============================================
-- Query 5: Head-to-head record between two teams
-- Business Question: What is the historical win-loss record between two specific teams?
-- Why this matters: Identifies dominant rivalries and can inform predictions
-- for upcoming matches between these teams
-- ============================================

WITH head_to_head AS (
    -- Get all matches between two teams
    SELECT 
        t1.team_name AS team1,
        t2.team_name AS team2,
        CASE 
            WHEN m.winner_id = m.team1_id THEN t1.team_name
            WHEN m.winner_id = m.team2_id THEN t2.team_name
            ELSE 'No Result'
        END AS winner,
        m.margin,
        m.match_date
    FROM matches m
    JOIN teams t1 ON m.team1_id = t1.team_id
    JOIN teams t2 ON m.team2_id = t2.team_id
    WHERE (t1.team_name = 'India' AND t2.team_name = 'Pakistan')  -- Replace with desired teams
       OR (t1.team_name = 'Pakistan' AND t2.team_name = 'India')
),
match_counts AS (
    -- Count wins for each team
    SELECT 
        team1,
        team2,
        SUM(CASE WHEN winner = team1 THEN 1 ELSE 0 END) AS team1_wins,
        SUM(CASE WHEN winner = team2 THEN 1 ELSE 0 END) AS team2_wins,
        SUM(CASE WHEN winner = 'No Result' THEN 1 ELSE 0 END) AS no_results,
        COUNT(*) AS total_matches
    FROM head_to_head
    GROUP BY team1, team2
)
SELECT 
    team1,
    team2,
    team1_wins,
    team2_wins,
    no_results,
    total_matches,
    ROUND(100.0 * team1_wins / NULLIF(total_matches, 0), 2) AS team1_win_percentage,
    ROUND(100.0 * team2_wins / NULLIF(total_matches, 0), 2) AS team2_win_percentage
FROM match_counts;

-- Also show recent match results between the teams
SELECT 
    team1,
    team2,
    winner,
    margin,
    match_date
FROM head_to_head
ORDER BY match_date DESC
LIMIT 10;
