"""
Predicting the outcome of the 2025/26 La Liga season using historical performance data.

This script takes historical La Liga seasons in CSV format, builds per-team
summary statistics with proper trend analysis, trains a regression model,
and predicts the final league positions for 2025/26.

The 2025/26 La Liga will include 20 clubs - the 17 sides that remained in
the division in 2024/25 and three promoted clubs (Levante, Oviedo, and Elche).

The historical match files used by this script should be in CSV format with
columns for Date, Team 1, FT (final score), HT (half-time score), and Team 2.
Each file should contain all matches for a given season.

Usage
-----
Run the script from a terminal with Python 3. Ensure that pandas, numpy
and scikit-learn are installed. All required CSV files should reside in
the same directory as this script.

Example:
    python3 laliga_2025-26_prediction.py

The script outputs a predicted ranking of the 20 clubs for the 2025/26
La Liga season.
"""

import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def parse_match_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse final score into integer goal columns, handling missing values properly.
    
    The raw CSV files use a `FT` column that stores the full-time result as a
    string such as "2-1". This helper splits the column into separate home and
    away goal counts and returns an updated DataFrame with `home_goals` and
    `away_goals` columns.

    Parameters
    ----------
    df : DataFrame
        Match data with columns Team 1, Team 2 and FT.

    Returns
    -------
    DataFrame
        DataFrame with added home_goals and away_goals columns.
    """
    df = df.copy()
    
    df = df.dropna(subset=["FT"])
    df["FT"] = df["FT"].astype(str).str.strip()
    df = df.drop_duplicates()
    
    def create_match_key(row):
        teams = sorted([row["Team 1"], row["Team 2"]])
        return f"{teams[0]} vs {teams[1]} - {row['FT']}"
    
    df["match_key"] = df.apply(create_match_key, axis=1)
    df = df.drop_duplicates(subset=["match_key"], keep="first")
    df = df.drop_duplicates(subset=["Team 1", "Team 2", "FT"], keep="first")
    
    goals = df["FT"].str.split("-", expand=True)
    df["home_goals"] = pd.to_numeric(goals[0], errors="coerce")
    df["away_goals"] = pd.to_numeric(goals[1], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df = df.drop("match_key", axis=1)
    
    return df


def calculate_team_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive team statistics for a season.
    
    Given a DataFrame of matches with columns Team 1, Team 2, home_goals
    and away_goals, compute the total points, wins, draws, losses, goals
    for and against, goal difference, and per-game statistics for each team.
    After accumulating statistics, the teams are sorted by points (descending),
    goal difference (descending) and goals for (descending) to determine the
    final ranking.

    Parameters
    ----------
    matches : DataFrame
        DataFrame of parsed match results.

    Returns
    -------
    DataFrame
        Summary of the season with one row per team and columns:
        [team, points, wins, draws, losses, goals_for, goals_against,
         goal_diff, matches_played, points_per_game, goals_per_game,
         goals_against_per_game, position].
    """
    teams = defaultdict(lambda: {
        "points": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "matches_played": 0
    })

    for _, row in matches.iterrows():
        home, away = row["Team 1"], row["Team 2"]
        hg, ag = row["home_goals"], row["away_goals"]
        
        teams[home]["goals_for"] += hg
        teams[home]["goals_against"] += ag
        teams[away]["goals_for"] += ag
        teams[away]["goals_against"] += hg
        
        teams[home]["matches_played"] += 1
        teams[away]["matches_played"] += 1
        
        if hg > ag:
            teams[home]["points"] += 3
            teams[home]["wins"] += 1
            teams[away]["losses"] += 1
        elif hg < ag:
            teams[away]["points"] += 3
            teams[away]["wins"] += 1
            teams[home]["losses"] += 1
        else:
            teams[home]["points"] += 1
            teams[away]["points"] += 1
            teams[home]["draws"] += 1
            teams[away]["draws"] += 1

    data = []
    for team, stats in teams.items():
        goal_diff = stats["goals_for"] - stats["goals_against"]
        points_per_game = stats["points"] / max(stats["matches_played"], 1)
        goals_per_game = stats["goals_for"] / max(stats["matches_played"], 1)
        goals_against_per_game = stats["goals_against"] / max(stats["matches_played"], 1)
        
        data.append({
            "team": team,
            "points": stats["points"],
            "wins": stats["wins"],
            "draws": stats["draws"],
            "losses": stats["losses"],
            "goals_for": stats["goals_for"],
            "goals_against": stats["goals_against"],
            "goal_diff": goal_diff,
            "matches_played": stats["matches_played"],
            "points_per_game": points_per_game,
            "goals_per_game": goals_per_game,
            "goals_against_per_game": goals_against_per_game
        })

    summary = pd.DataFrame(data)
    summary = summary.sort_values(
        ["points", "goal_diff", "goals_for"], 
        ascending=[False, False, False]
    ).reset_index(drop=True)
    
    summary["position"] = summary.index + 1
    return summary


def create_team_features(team_name: str, season_summaries: Dict[str, pd.DataFrame], 
                        current_season: str) -> Dict[str, float]:
    """
    Create comprehensive features for a team based on historical performance.
    
    Parameters
    ----------
    team_name : str
        Name of the team to create features for.
    season_summaries : Dict[str, pd.DataFrame]
        Dictionary mapping season file paths to season summaries.
    current_season : str
        The season file path to use as the current season.
        
    Returns
    -------
    Dict[str, float]
        Dictionary of features for the team, or empty dict if team not found.
    """
    
    # Get all seasons up to the current one
    seasons = [s for s in season_summaries.keys() if s <= current_season]
    seasons.sort()
    
    if not seasons:
        return {}
    
    # Get the most recent season summary
    latest_summary = season_summaries[seasons[-1]]
    
    if team_name in latest_summary["team"].values:
        team_stats = latest_summary[latest_summary["team"] == team_name].iloc[0]
        
        # Create features - only include the ones used in training
        features = {
            "points": team_stats["points"],
            "wins": team_stats["wins"],
            "draws": team_stats["draws"],
            "losses": team_stats["losses"],
            "goals_for": team_stats["goals_for"],
            "goals_against": team_stats["goals_against"],
            "goal_diff": team_stats["goal_diff"],
            "points_per_game": team_stats["points_per_game"],
            "goals_per_game": team_stats["goals_per_game"],
            "goals_against_per_game": team_stats["goals_against_per_game"]
        }
        
        # Add trend features if we have multiple seasons
        if len(seasons) >= 2:
            prev_season = seasons[-2]
            if team_name in season_summaries[prev_season]["team"].values:
                prev_stats = season_summaries[prev_season][season_summaries[prev_season]["team"] == team_name].iloc[0]
                
                features.update({
                    "points_change": team_stats["points"] - prev_stats["points"],
                    "goals_for_change": team_stats["goals_for"] - prev_stats["goals_for"],
                    "goals_against_change": team_stats["goals_against"] - prev_stats["goals_against"],
                    "goal_diff_change": team_stats["goal_diff"] - prev_stats["goal_diff"]
                })
            else:
                # Team wasn't in previous season (promoted)
                features.update({
                    "points_change": 0,
                    "goals_for_change": 0,
                    "goals_against_change": 0,
                    "goal_diff_change": 0
                })
        else:
            features.update({
                "points_change": 0,
                "goals_for_change": 0,
                "goals_against_change": 0,
                "goal_diff_change": 0
            })
        
        return features
    
    return {}


def debug_team_creation(team_name: str, season_summaries: Dict[str, pd.DataFrame]):
    """Debug function to see what's happening with team feature creation."""
    print(f"\nDEBUG: Creating features for {team_name}")
    for season_file, summary in season_summaries.items():
        if team_name in summary["team"].values:
            team_data = summary[summary["team"] == team_name]
            print(f"  Found in {season_file}: {len(team_data)} rows")
            for idx, row in team_data.iterrows():
                print(f"    Row {idx}: {dict(row)}")
        else:
            print(f"  Not found in {season_file}")


def prepare_training_data(season_files: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepare training features and labels from a list of seasons.
    
    Given a list of file paths ordered chronologically, compute per-team
    statistics for each season and build a dataset where the feature
    vector for season n+1 comes from the statistics of season n.
    Teams promoted into La Liga without previous season statistics are
    assigned default feature values equal to the average of the bottom
    three clubs in the prior season.

    Parameters
    ----------
    season_files : List[str]
        Paths to season CSV files ordered from oldest to newest.

    Returns
    -------
    X_train : DataFrame
        Feature matrix (numeric) for training.
    y_train : Series
        Target series containing league positions (1-20).
    latest_features : DataFrame
        Feature matrix for the most recent season in the list (used
        for prediction).
    """
    
    season_summaries = {}
    
    # Process each season file
    for file_path in season_files:
        try:
            raw = pd.read_csv(file_path)
            parsed = parse_match_results(raw)
            summary = calculate_team_stats(parsed)
            season_summaries[file_path] = summary
        except Exception as e:
            print(f"Warning: Could not process {file_path}: {e}")
            continue
    
    if not season_summaries:
        raise ValueError("No valid season data found")
    
    # Create training data
    feature_rows = []
    target_rows = []
    
    # Sort seasons chronologically
    sorted_seasons = sorted(season_summaries.keys())
    
    for i in range(len(sorted_seasons) - 1):
        current_season = sorted_seasons[i + 1]
        current_summary = season_summaries[current_season]
        
        for _, team_row in current_summary.iterrows():
            team_name = team_row["team"]
            features = create_team_features(team_name, season_summaries, sorted_seasons[i])
            
            if features:
                feature_rows.append(features)
                target_rows.append(team_row["position"])
    
    if not feature_rows:
        raise ValueError("No training data could be created")
    
    X_train = pd.DataFrame(feature_rows)
    y_train = pd.Series(target_rows)
    
    # Get the most recent season for prediction
    latest_season = sorted_seasons[-1]
    latest_summary = season_summaries[latest_season]
    
    # Remove relegated teams (bottom 3)
    relegated = latest_summary.tail(3)["team"].tolist()
    latest_summary = latest_summary[~latest_summary["team"].isin(relegated)]
    
    # Ensure no duplicate teams (this should already be handled, but double-check)
    latest_summary = latest_summary.drop_duplicates(subset=["team"], keep="first")
    
    # Define the actual teams that should be in La Liga 2025/26
    # These are the teams that remained in La Liga after 2024/25 + the 3 promoted teams
    # Removed: Leganes, Las Palmas, Valladolid (relegated)
    la_liga_2025_26_teams = [
        "Real Madrid", "Barcelona", "Ath Madrid", "Ath Bilbao", "Villarreal",
        "Girona", "Betis", "Sociedad", "Valencia", "Getafe",
        "Osasuna", "Sevilla", "Celta", "Mallorca", "Alaves", "Vallecano",
        "Espanyol", "Levante", "Oviedo", "Elche"
    ]
    
    # Filter latest_summary to only include teams that should be in La Liga 2025/26
    latest_summary = latest_summary[latest_summary["team"].isin(la_liga_2025_26_teams)]
    
    # Add promoted teams with default features
    promoted = ["Levante", "Oviedo", "Elche"]
    
    # Calculate average features for bottom teams as baseline for promoted teams
    bottom_teams = latest_summary.tail(3)
    default_features = {}
    
    # Create default features matching the training feature structure
    for col in ["points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"]:
        default_features[col] = bottom_teams[col].mean() * 0.7  # Worse than bottom teams
    
    for col in ["points_per_game", "goals_per_game", "goals_against_per_game"]:
        default_features[col] = bottom_teams[col].mean() * 0.75
    
    # Add trend features (set to 0 for new teams)
    for col in ["points_change", "goals_for_change", "goals_against_change", "goal_diff_change"]:
        default_features[col] = 0
    
    # Add promoted teams - only if they don't already exist
    for team in promoted:
        if team not in latest_summary["team"].values:
            # Use default features for new teams
            team_features = default_features.copy()
            team_features["team"] = team
            
            # Add to latest summary
            new_row = pd.DataFrame([team_features])
            latest_summary = pd.concat([latest_summary, new_row], ignore_index=True)
    
    # Final deduplication to ensure no duplicates remain
    latest_summary = latest_summary.drop_duplicates(subset=["team"], keep="first")
    
    # We should now have exactly 20 teams (17 existing + 3 promoted)
    if len(latest_summary) != 20:
        print(f"WARNING: Expected 20 teams, but got {len(latest_summary)}")
        print(f"Teams found: {sorted(latest_summary['team'].tolist())}")
        
        # If we have more than 20, truncate to the top teams
        if len(latest_summary) > 20:
            existing_teams = latest_summary[~latest_summary['team'].isin(promoted)].head(17)
            promoted_teams = latest_summary[latest_summary['team'].isin(promoted)]
            latest_summary = pd.concat([existing_teams, promoted_teams]).reset_index(drop=True)
    
    # Debug: Print team count and check for duplicates
    print(f"Final team count: {len(latest_summary)}")
    print(f"Unique teams: {len(latest_summary['team'].unique())}")
    if len(latest_summary) != len(latest_summary['team'].unique()):
        print("WARNING: Duplicate teams detected!")
        duplicate_counts = latest_summary['team'].value_counts()
        duplicates = duplicate_counts[duplicate_counts > 1]
        print(f"Duplicate teams: {list(duplicates.index)}")
        print("Full duplicate details:")
        for team in duplicates.index:
            team_rows = latest_summary[latest_summary['team'] == team]
            print(f"  {team}: {len(team_rows)} rows")
            for idx, row in team_rows.iterrows():
                print(f"    Row {idx}: {dict(row)}")
        
        # Debug the first duplicate team to see where it's coming from
        if duplicates.index:
            first_duplicate = duplicates.index[0]
            debug_team_creation(first_duplicate, season_summaries)
    
    return X_train, y_train, latest_summary


def build_and_train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """
    Create a pipeline that scales features and trains a RandomForest.
    
    Parameters
    ----------
    X : DataFrame
        Training features.
    y : Series
        Target positions (1-20).

    Returns
    -------
    Pipeline
        Scikit-learn pipeline with StandardScaler and RandomForestRegressor.
    """
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    
    # Train the model
    model.fit(X, y)
    
    print(f"Model trained with {X.shape[1]} features")
    return model


def predict_league_table(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """
    Predict the league table ordering for the given features.
    
    Parameters
    ----------
    model : Pipeline
        Trained scikit-learn pipeline.
    features : DataFrame
        Feature rows indexed by team name.

    Returns
    -------
    DataFrame
        Predicted positions sorted from 1 to 20.
    """
    
    features = features.drop_duplicates(subset=["team"], keep="first").reset_index(drop=True)
    feature_cols = [col for col in features.columns if col not in ["team", "position", "matches_played"]]
    X_pred = features[feature_cols].fillna(0)
    
    probas = model.predict_proba(X_pred)
    classes = np.arange(1, 21)
    expected_positions = np.sum(probas * classes, axis=1)
    
    # Create prediction DataFrame
    predictions = pd.DataFrame({
        "team": features["team"],
        "predicted_position": expected_positions
    })
    
    predictions = predictions.sort_values("predicted_position").reset_index(drop=True)
    predictions["predicted_rank"] = predictions.index + 1
    
    promoted = ["Levante", "Oviedo", "Elche"]
    promoted_set = set(promoted)
    
    normal_teams = predictions[~predictions['team'].isin(promoted_set)].copy()
    promoted_teams = predictions[predictions['team'].isin(promoted_set)].copy()
    
    promoted_teams = promoted_teams.sort_values("predicted_position")
    
    for i, (_, team_row) in enumerate(promoted_teams.iterrows()):
        promoted_teams.loc[team_row.name, "predicted_position"] = 18 + i
    
    final_predictions = pd.concat([normal_teams, promoted_teams]).reset_index(drop=True)
    final_predictions = final_predictions.sort_values("predicted_position").reset_index(drop=True)
    final_predictions["final_rank"] = final_predictions.index + 1
    
    final_predictions = final_predictions.head(20)
    
    return final_predictions[["final_rank", "team", "predicted_position"]]


def predict_match_result(model: Pipeline, features: pd.DataFrame, home_team: str, away_team: str) -> Dict:
    """
    Predict the outcome of a specific match between two teams.
    
    Parameters
    ----------
    model : Pipeline
        Trained scikit-learn pipeline.
    features : DataFrame
        Feature rows indexed by team name.
    home_team : str
        Name of the home team.
    away_team : str
        Name of the away team.

    Returns
    -------
    Dict
        Dictionary containing match prediction details.
    """
    
    if home_team not in features["team"].values or away_team not in features["team"].values:
        return {"error": "One or both teams not found in features"}
    
    home_features = features[features["team"] == home_team].iloc[0]
    away_features = features[features["team"] == away_team].iloc[0]
    
    feature_cols = [col for col in features.columns if col not in ["team", "position", "matches_played"]]
    
    home_X = home_features[feature_cols].fillna(0).values.reshape(1, -1)
    away_X = away_features[feature_cols].fillna(0).values.reshape(1, -1)
    
    home_probas = model.predict_proba(home_X)[0]
    away_probas = model.predict_proba(away_X)[0]
    
    home_position = np.sum(home_probas * np.arange(1, 21))
    away_position = np.sum(away_probas * np.arange(1, 21))
    
    position_diff = home_position - away_position
    
    home_goals = max(1, round(2.5 - (home_position - 1) * 0.1))
    away_goals = max(0, round(1.5 - (away_position - 1) * 0.1))
    
    if position_diff < -2:
        win_prob = 0.15
        draw_prob = 0.25
        loss_prob = 0.60
    elif position_diff < 0:
        win_prob = 0.30
        draw_prob = 0.35
        loss_prob = 0.35
    elif position_diff < 2:
        win_prob = 0.45
        draw_prob = 0.30
        loss_prob = 0.25
    else:
        win_prob = 0.60
        draw_prob = 0.25
        loss_prob = 0.15
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_position": round(home_position, 1),
        "away_position": round(away_position, 1),
        "position_difference": round(position_diff, 1),
        "predicted_score": f"{home_goals}-{away_goals}",
        "win_probability": round(win_prob * 100, 1),
        "draw_probability": round(draw_prob * 100, 1),
        "loss_probability": round(loss_prob * 100, 1),
        "predicted_outcome": "Home Win" if win_prob > 0.4 else "Draw" if draw_prob > 0.35 else "Away Win"
    }


def main():
    """
    Main execution function.
    
    Defines season files, prepares training data, trains the model,
    and outputs predictions for the 2025/26 La Liga season.
    """
    
    # Define season files in chronological order
    season_files = [
        "laliga_2019_2020.csv",
        "laliga_2020_2021.csv",
        "laliga_2021_2022.csv",
        "laliga_2022_2023.csv",
        "laliga_2023_2024.csv",
        "laliga_2024_2025.csv"
    ]
    
    # Check which files exist
    existing_files = [f for f in season_files if os.path.exists(f)]
    
    if len(existing_files) < 2:
        print("Error: Need at least 2 season files to make predictions")
        return
    
    print(f"Using {len(existing_files)} season files for prediction")
    
    try:
        # Prepare data
        X_train, y_train, latest_features = prepare_training_data(existing_files)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Features available: {list(X_train.columns)}")
        
        # Train model
        model = build_and_train_model(X_train, y_train)
        
        # Make predictions
        predictions = predict_league_table(model, latest_features)
        
        # Debug: Check for duplicates in final predictions
        print(f"\nFinal prediction team count: {len(predictions)}")
        print(f"Unique teams in predictions: {len(predictions['team'].unique())}")
        if len(predictions) != len(predictions['team'].unique()):
            print("WARNING: Duplicates in final predictions!")
            duplicate_counts = predictions['team'].value_counts()
            duplicates = duplicate_counts[duplicate_counts > 1]
            print(f"Duplicate teams: {list(duplicates.index)}")
        
        # Display results
        print("\n" + "="*60)
        print("PREDICTED LA LIGA 2025/26 TABLE")
        print("="*60)
        print(f"{'Rank':<4} {'Team':<20} {'Predicted Position':<20}")
        print("-" * 60)
        
        for _, row in predictions.iterrows():
            rank = int(row['final_rank'])
            team = row['team']
            pred_pos = row['predicted_position']
            
            # Highlight promoted teams
            if team in ["Levante", "Oviedo", "Elche"]:
                print(f"{rank:<4} {team:<20} {pred_pos:<20.1f} (P)")
            else:
                print(f"{rank:<4} {team:<20} {pred_pos:<20.1f}")
        
        print("\n(P) = Promoted team")
        print("="*60)
        
        print("\n" + "="*60)
        print("MATCH PREDICTIONS")
        print("="*60)
        
        sample_matches = [
            ("Elche", "Levante"),
            ("Valencia", "Getafe"),
            ("Alaves", "Ath Madrid"),
            ("Oviedo", "Sociedad"),
            ("Girona", "Sevilla")
        ]
        
        for home, away in sample_matches:
            match_pred = predict_match_result(model, latest_features, home, away)
            if "error" not in match_pred:
                print(f"\n{home} vs {away}")
                print(f"Predicted Score: {match_pred['predicted_score']}")
                print(f"Outcome: {match_pred['predicted_outcome']}")
                print(f"Win: {match_pred['win_probability']}% | Draw: {match_pred['draw_probability']}% | Loss: {match_pred['loss_probability']}%")
                print(f"Position Diff: {match_pred['position_difference']}")
        
        print("\n" + "="*60)
        
        print("\nTo predict a specific match, use:")
        print("predict_specific_match('Home Team', 'Away Team')")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()


def predict_specific_match(home_team: str, away_team: str):
    """
    Predict a specific match between two teams.
    
    Parameters
    ----------
    home_team : str
        Name of the home team.
    away_team : str
        Name of the away team.
    """
    try:
        season_files = [
            "laliga_2019_2020.csv",
            "laliga_2020_2021.csv", 
            "laliga_2021_2022.csv",
            "laliga_2022_2023.csv",
            "laliga_2023_2024.csv",
            "laliga_2024_2025.csv"
        ]
        
        existing_files = [f for f in season_files if os.path.exists(f)]
        if len(existing_files) < 2:
            print("Error: Need at least 2 season files to make predictions")
            return
        
        X_train, y_train, latest_features = prepare_training_data(existing_files)
        model = build_and_train_model(X_train, y_train)
        
        match_pred = predict_match_result(model, latest_features, home_team, away_team)
        
        if "error" in match_pred:
            print(f"Error: {match_pred['error']}")
            return
        
        print(f"\n{'='*50}")
        print(f"MATCH PREDICTION: {home_team} vs {away_team}")
        print(f"{'='*50}")
        print(f"Predicted Score: {match_pred['predicted_score']}")
        print(f"Predicted Outcome: {match_pred['predicted_outcome']}")
        print(f"Win Probability: {match_pred['win_probability']}%")
        print(f"Draw Probability: {match_pred['draw_probability']}%") 
        print(f"Loss Probability: {match_pred['loss_probability']}%")
        print(f"Position Difference: {match_pred['position_difference']}")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error predicting match: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()