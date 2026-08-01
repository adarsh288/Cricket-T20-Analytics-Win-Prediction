"""
Core scraping logic for ESPN Cricinfo T20 data.
Modular functions for different data types: batting, bowling, player info, match results.
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import logging

from utils import retry_request, log_progress, HEADERS

logger = logging.getLogger(__name__)


def get_match_links(tournament_url: str) -> List[str]:
    """
    Get all match scorecard links from a tournament page.
    
    Why this structure: This is a helper function used by multiple scrapers.
    Extracting it separately avoids code duplication - we need match links for
    batting, bowling, and player info scraping.
    
    Args:
        tournament_url: URL of the tournament match results page
        
    Returns:
        List of full URLs to individual match scorecards
    """
    def make_request():
        response = requests.get(tournament_url, headers=HEADERS)
        response.raise_for_status()
        return response
    
    response = retry_request(make_request)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    links = []
    # Find all rows in the match results table
    # Using the same selector as original: table.engineTable > tbody > tr.data1
    rows = soup.select('table.engineTable > tbody > tr.data1')
    
    for row in rows:
        tds = row.find_all('td')
        if len(tds) >= 7:
            # Get the scorecard link from column 7 (index 6)
            link_tag = tds[6].find('a')
            if link_tag and link_tag.get('href'):
                full_url = "https://www.espncricinfo.com" + link_tag['href']
                links.append(full_url)
    
    logger.info(f"Found {len(links)} match links")
    return links


def scrape_batting_summary(tournament_url: str) -> List[Dict]:
    """
    Scrape batting summary for all matches in a tournament.
    
    Why this structure: 
    - Separated into two logical steps: get links, then scrape each match
    - Each match produces batting data for both innings
    - Returns a flat list of all batting records (easy to convert to DataFrame later)
    
    Args:
        tournament_url: URL of the tournament match results page
        
    Returns:
        List of dictionaries, each containing batting stats for one player in one innings
    """
    match_links = get_match_links(tournament_url)
    all_batting_data = []
    
    for idx, match_url in enumerate(match_links):
        log_progress(idx + 1, len(match_links), "matches")
        
        def make_request():
            response = requests.get(match_url, headers=HEADERS)
            response.raise_for_status()
            return response
        
        try:
            response = retry_request(make_request)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract team names from match details section
            # Original logic: find div with "Match Details" span, get siblings
            match_divs = soup.find_all('div')
            match_details_div = None
            for div in match_divs:
                span = div.find('span')
                if span and span.find('span'):
                    if span.find('span').text == "Match Details":
                        match_details_div = div
                        break
            
            if match_details_div:
                siblings = match_details_div.find_next_siblings('div')
                if len(siblings) >= 2:
                    team1 = siblings[0].find('span').find('span').text.replace(" Innings", "")
                    team2 = siblings[1].find('span').find('span').text.replace(" Innings", "")
                    match_info = f"{team1} Vs {team2}"
                else:
                    logger.warning(f"Could not extract team names from {match_url}")
                    continue
            else:
                logger.warning(f"Could not find match details section in {match_url}")
                continue
            
            # Find batting scorecard tables
            # Original: div > table.ci-scorecard-table
            tables = soup.select('div > table.ci-scorecard-table')
            
            if len(tables) < 2:
                logger.warning(f"Expected 2 innings tables, found {len(tables)} in {match_url}")
                continue
            
            # Process first innings
            first_innings_rows = tables[0].select('tbody > tr')
            for row in first_innings_rows:
                tds = row.find_all('td')
                if len(tds) >= 8:
                    batting_data = {
                        "match": match_info,
                        "teamInnings": team1,
                        "battingPos": len(all_batting_data) + 1,  # Will be recalculated per innings
                        "batsmanName": tds[0].find('a').find('span').find('span').text.replace('\xa0', ''),
                        "dismissal": tds[1].find('span').find('span').text,
                        "runs": tds[2].find('strong').text,
                        "balls": tds[3].text,
                        "4s": tds[5].text,
                        "6s": tds[6].text,
                        "SR": tds[7].text
                    }
                    all_batting_data.append(batting_data)
            
            # Process second innings
            second_innings_rows = tables[1].select('tbody > tr')
            for row in second_innings_rows:
                tds = row.find_all('td')
                if len(tds) >= 8:
                    batting_data = {
                        "match": match_info,
                        "teamInnings": team2,
                        "battingPos": len(all_batting_data) + 1,
                        "batsmanName": tds[0].find('a').find('span').find('span').text.replace('\xa0', ''),
                        "dismissal": tds[1].find('span').find('span').text,
                        "runs": tds[2].find('strong').text,
                        "balls": tds[3].text,
                        "4s": tds[5].text,
                        "6s": tds[6].text,
                        "SR": tds[7].text
                    }
                    all_batting_data.append(batting_data)
            
        except Exception as e:
            logger.error(f"Error scraping batting data from {match_url}: {e}")
            continue
    
    # Recalculate batting positions per innings
    # Why: The original logic calculated position per innings, not globally
    current_match = None
    current_innings = None
    position_counter = 1
    
    for record in all_batting_data:
        if record['match'] != current_match or record['teamInnings'] != current_innings:
            current_match = record['match']
            current_innings = record['teamInnings']
            position_counter = 1
        record['battingPos'] = position_counter
        position_counter += 1
    
    logger.info(f"Scraped {len(all_batting_data)} batting records")
    return all_batting_data


def scrape_bowling_summary(tournament_url: str) -> List[Dict]:
    """
    Scrape bowling summary for all matches in a tournament.
    
    Why this structure:
    - Similar to batting summary but extracts bowling-specific tables
    - Bowling team is the OPPOSITE of the batting team (they bowl against each other)
    - Uses different table selector (ds-table vs ci-scorecard-table)
    
    Args:
        tournament_url: URL of the tournament match results page
        
    Returns:
        List of dictionaries, each containing bowling stats for one bowler in one innings
    """
    match_links = get_match_links(tournament_url)
    all_bowling_data = []
    
    for idx, match_url in enumerate(match_links):
        log_progress(idx + 1, len(match_links), "matches")
        
        def make_request():
            response = requests.get(match_url, headers=HEADERS)
            response.raise_for_status()
            return response
        
        try:
            response = retry_request(make_request)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract team names (same logic as batting)
            match_divs = soup.find_all('div')
            match_details_div = None
            for div in match_divs:
                span = div.find('span')
                if span and span.find('span'):
                    if span.find('span').text == "Match Details":
                        match_details_div = div
                        break
            
            if match_details_div:
                siblings = match_details_div.find_next_siblings('div')
                if len(siblings) >= 2:
                    team1 = siblings[0].find('span').find('span').text.replace(" Innings", "")
                    team2 = siblings[1].find('span').find('span').text.replace(" Innings", "")
                    match_info = f"{team1} Vs {team2}"
                else:
                    logger.warning(f"Could not extract team names from {match_url}")
                    continue
            else:
                logger.warning(f"Could not find match details section in {match_url}")
                continue
            
            # Find bowling tables
            # Original: div > table.ds-table, tables at indices 1 and 3
            tables = soup.select('div > table.ds-table')
            
            if len(tables) < 4:
                logger.warning(f"Expected 4 ds-table elements, found {len(tables)} in {match_url}")
                continue
            
            # Process first innings bowling (table index 1)
            # Bowling team is team2 (they bowl against team1's batting)
            first_innings_rows = tables[1].select('tbody > tr')
            for row in first_innings_rows:
                tds = row.find_all('td')
                if len(tds) >= 11:
                    bowling_data = {
                        "match": match_info,
                        "bowlingTeam": team2,
                        "bowlerName": tds[0].find('a').find('span').text.replace('\xa0', ''),
                        "overs": tds[1].text,
                        "maiden": tds[2].text,
                        "runs": tds[3].text,
                        "wickets": tds[4].text,
                        "economy": tds[5].text,
                        "0s": tds[6].text,
                        "4s": tds[7].text,
                        "6s": tds[8].text,
                        "wides": tds[9].text,
                        "noBalls": tds[10].text
                    }
                    all_bowling_data.append(bowling_data)
            
            # Process second innings bowling (table index 3)
            # Bowling team is team1 (they bowl against team2's batting)
            second_innings_rows = tables[3].select('tbody > tr')
            for row in second_innings_rows:
                tds = row.find_all('td')
                if len(tds) >= 11:
                    bowling_data = {
                        "match": match_info,
                        "bowlingTeam": team1,
                        "bowlerName": tds[0].find('a').find('span').text.replace('\xa0', ''),
                        "overs": tds[1].text,
                        "maiden": tds[2].text,
                        "runs": tds[3].text,
                        "wickets": tds[4].text,
                        "economy": tds[5].text,
                        "0s": tds[6].text,
                        "4s": tds[7].text,
                        "6s": tds[8].text,
                        "wides": tds[9].text,
                        "noBalls": tds[10].text
                    }
                    all_bowling_data.append(bowling_data)
            
        except Exception as e:
            logger.error(f"Error scraping bowling data from {match_url}: {e}")
            continue
    
    logger.info(f"Scraped {len(all_bowling_data)} bowling records")
    return all_bowling_data


def scrape_player_info(tournament_url: str) -> List[Dict]:
    """
    Scrape player information (batting style, bowling style, playing role, description).
    
    Why this structure:
    - Three-stage process: get match links → extract player links from matches → scrape each player page
    - This is necessary because player info is on individual player profile pages, not match pages
    - Deduplication is needed since same player appears in multiple matches
    
    Args:
        tournament_url: URL of the tournament match results page
        
    Returns:
        List of dictionaries with player profile information
    """
    match_links = get_match_links(tournament_url)
    all_player_links = []  # Will store (name, team, profile_url) tuples
    
    # Stage 1: Collect all unique player profile links from all matches
    for idx, match_url in enumerate(match_links):
        log_progress(idx + 1, len(match_links), "matches (collecting player links)")
        
        def make_request():
            response = requests.get(match_url, headers=HEADERS)
            response.raise_for_status()
            return response
        
        try:
            response = retry_request(make_request)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract team names
            match_divs = soup.find_all('div')
            match_details_div = None
            for div in match_divs:
                span = div.find('span')
                if span and span.find('span'):
                    if span.find('span').text == "Match Details":
                        match_details_div = div
                        break
            
            if not match_details_div:
                continue
            
            siblings = match_details_div.find_next_siblings('div')
            if len(siblings) < 2:
                continue
            
            team1 = siblings[0].find('span').find('span').text.replace(" Innings", "")
            team2 = siblings[1].find('span').find('span').text.replace(" Innings", "")
            
            # Extract batting player links
            batting_tables = soup.select('div > table.ci-scorecard-table')
            if len(batting_tables) >= 2:
                for table_idx, table in enumerate(batting_tables[:2]):
                    team = team1 if table_idx == 0 else team2
                    rows = table.select('tbody > tr')
                    for row in rows:
                        tds = row.find_all('td')
                        if len(tds) >= 8:
                            link_tag = tds[0].find('a')
                            if link_tag and link_tag.get('href'):
                                name = link_tag.find('span').find('span').text.replace('\xa0', '')
                                profile_url = "https://www.espncricinfo.com" + link_tag['href']
                                all_player_links.append((name, team, profile_url))
            
            # Extract bowling player links
            bowling_tables = soup.select('div > table.ds-table')
            if len(bowling_tables) >= 4:
                # First innings bowling (table index 1) - team2 bowlers
                rows = bowling_tables[1].select('tbody > tr')
                for row in rows:
                    tds = row.find_all('td')
                    if len(tds) >= 11:
                        link_tag = tds[0].find('a')
                        if link_tag and link_tag.get('href'):
                            name = link_tag.find('span').text.replace('\xa0', '')
                            profile_url = "https://www.espncricinfo.com" + link_tag['href']
                            all_player_links.append((name, team2, profile_url))
                
                # Second innings bowling (table index 3) - team1 bowlers
                rows = bowling_tables[3].select('tbody > tr')
                for row in rows:
                    tds = row.find_all('td')
                    if len(tds) >= 11:
                        link_tag = tds[0].find('a')
                        if link_tag and link_tag.get('href'):
                            name = link_tag.find('span').text.replace('\xa0', '')
                            profile_url = "https://www.espncricinfo.com" + link_tag['href']
                            all_player_links.append((name, team1, profile_url))
            
        except Exception as e:
            logger.error(f"Error collecting player links from {match_url}: {e}")
            continue
    
    # Deduplicate player links (same player may appear in multiple matches)
    # Why: We only need to scrape each player's profile once
    unique_player_links = list(set(all_player_links))
    logger.info(f"Found {len(unique_player_links)} unique players")
    
    # Stage 2: Scrape each player's profile page
    all_player_data = []
    
    for idx, (name, team, profile_url) in enumerate(unique_player_links):
        log_progress(idx + 1, len(unique_player_links), "player profiles")
        
        def make_request():
            response = requests.get(profile_url, headers=HEADERS)
            response.raise_for_status()
            return response
        
        try:
            response = retry_request(make_request)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract batting style
            batting_style = ""
            batting_div = soup.select_one('div.ds-grid > div')
            for div in soup.select('div.ds-grid > div'):
                p_tag = div.find('p')
                if p_tag and p_tag.text == 'Batting Style':
                    span = div.find('span')
                    if span:
                        batting_style = span.text
                    break
            
            # Extract bowling style
            bowling_style = ""
            for div in soup.select('div.ds-grid > div'):
                p_tag = div.find('p')
                if p_tag and p_tag.text == 'Bowling Style':
                    span = div.find('span')
                    if span:
                        bowling_style = span.text
                    break
            
            # Extract playing role
            playing_role = ""
            for div in soup.select('div.ds-grid > div'):
                p_tag = div.find('p')
                if p_tag and p_tag.text == 'Playing Role':
                    span = div.find('span')
                    if span:
                        playing_role = span.text
                    break
            
            # Extract description/bio
            description = ""
            bio_div = soup.select_one('div.ci-player-bio-content')
            if bio_div:
                p_tag = bio_div.find('p')
                if p_tag:
                    description = p_tag.text
            
            player_data = {
                "name": name,
                "team": team,
                "battingStyle": batting_style,
                "bowlingStyle": bowling_style,
                "playingRole": playing_role,
                "description": description
            }
            all_player_data.append(player_data)
            
        except Exception as e:
            logger.error(f"Error scraping player data for {name} from {profile_url}: {e}")
            continue
    
    logger.info(f"Scraped {len(all_player_data)} player profiles")
    return all_player_data


def scrape_match_results(tournament_url: str) -> List[Dict]:
    """
    Scrape match results summary from tournament page.
    
    Why this structure:
    - Simplest of all scrapers - only needs the tournament page, not individual match pages
    - Direct table extraction from the main results page
    - Returns one record per match with basic match info
    
    Args:
        tournament_url: URL of the tournament match results page
        
    Returns:
        List of dictionaries with match summary information
    """
    def make_request():
        response = requests.get(tournament_url, headers=HEADERS)
        response.raise_for_status()
        return response
    
    response = retry_request(make_request)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    match_summary = []
    
    # Select all rows from the match results table
    rows = soup.select('table.engineTable > tbody > tr.data1')
    
    for row in rows:
        tds = row.find_all('td')
        if len(tds) >= 7:
            match_data = {
                'team1': tds[0].text,
                'team2': tds[1].text,
                'winner': tds[2].text,
                'margin': tds[3].text,
                'ground': tds[4].text,
                'matchDate': tds[5].text,
                'scorecard': tds[6].text
            }
            match_summary.append(match_data)
    
    logger.info(f"Scraped {len(match_summary)} match results")
    return match_summary


if __name__ == "__main__":
    import pandas as pd
    import os

    # Use script-relative paths to ensure data/raw is found regardless of CWD
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    RAW_DIR = os.path.join(PROJECT_ROOT, 'data/raw')
    
    # Create data/raw directory if it doesn't exist
    os.makedirs(RAW_DIR, exist_ok=True)

    # Tournament URL to scrape - replace with actual URL
    tournament_url = "https://www.espncricinfo.com/series/icc-men-s-t20-world-cup-2022-23-1298134/match-results"

    print("Scraping match results...")
    match_data = scrape_match_results(tournament_url)
    pd.DataFrame(match_data).to_csv(os.path.join(RAW_DIR, "match_results.csv"), index=False)
    print(f"Saved {len(match_data)} match results")

    print("Scraping batting summary...")
    batting_data = scrape_batting_summary(tournament_url)
    pd.DataFrame(batting_data).to_csv(os.path.join(RAW_DIR, "batting_summary.csv"), index=False)
    print(f"Saved {len(batting_data)} batting records")

    print("Scraping bowling summary...")
    bowling_data = scrape_bowling_summary(tournament_url)
    pd.DataFrame(bowling_data).to_csv(os.path.join(RAW_DIR, "bowling_summary.csv"), index=False)
    print(f"Saved {len(bowling_data)} bowling records")

    print("Scraping player info...")
    player_data = scrape_player_info(tournament_url)
    pd.DataFrame(player_data).to_csv(os.path.join(RAW_DIR, "player_info.csv"), index=False)
    print(f"Saved {len(player_data)} player records")

    print(f"Done. All files saved to {RAW_DIR}/")
