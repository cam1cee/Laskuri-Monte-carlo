import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime

class FootballPredictor:
    def __init__(self):
        self.over_under_model = None
        self.btts_model = None
        self.corners_model = None
        self.cards_model = None
        self.data_cache = {}
        
    def load_data(self, league_code, seasons):
        """Load data from football-data.co.uk"""
        all_data = []
        
        for season in seasons:
            try:
                url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
                df = pd.read_csv(url, encoding='latin1', on_bad_lines='skip')
                
                if not df.empty:
                    df['Season'] = season
                    all_data.append(df)
                    print(f"✓ Loaded {league_code} {season}: {len(df)} matches")
            except Exception as e:
                print(f"✗ Failed to load {league_code} {season}")
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    def prepare_features(self, df):
        """Engineer features for ML models"""
        df = df.copy()
        
        # Required columns check
        required = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        if not all(col in df.columns for col in required):
            return df
        
        # Calculate team statistics
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        
        features_list = []
        
        for idx, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            # Get historical stats before this match
            past_data = df[:idx]
            
            # Home team stats
            home_games = past_data[(past_data['HomeTeam'] == home_team) | (past_data['AwayTeam'] == home_team)]
            home_last5 = home_games.tail(5)
            
            # Away team stats
            away_games = past_data[(past_data['HomeTeam'] == away_team) | (past_data['AwayTeam'] == away_team)]
            away_last5 = away_games.tail(5)
            
            # Calculate features
            features = {
                # Home team features
                'home_goals_avg': self._get_goals_scored(past_data, home_team),
                'home_conceded_avg': self._get_goals_conceded(past_data, home_team),
                'home_form': self._get_form(home_last5, home_team),
                
                # Away team features
                'away_goals_avg': self._get_goals_scored(past_data, away_team),
                'away_conceded_avg': self._get_goals_conceded(past_data, away_team),
                'away_form': self._get_form(away_last5, away_team),
                
                # Additional features if available
                'home_corners_avg': self._get_corners(past_data, home_team) if 'HC' in df.columns else 0,
                'away_corners_avg': self._get_corners(past_data, away_team) if 'AC' in df.columns else 0,
                'home_cards_avg': self._get_cards(past_data, home_team) if 'HY' in df.columns else 0,
                'away_cards_avg': self._get_cards(past_data, away_team) if 'AY' in df.columns else 0,
            }
            
            # Targets
            if pd.notna(row['FTHG']) and pd.notna(row['FTAG']):
                total_goals = row['FTHG'] + row['FTAG']
                features['over_2_5'] = 1 if total_goals > 2.5 else 0
                features['btts'] = 1 if row['FTHG'] > 0 and row['FTAG'] > 0 else 0
                
                if 'HC' in df.columns and 'AC' in df.columns:
                    total_corners = row.get('HC', 0) + row.get('AC', 0)
                    features['over_10_5_corners'] = 1 if total_corners > 10.5 else 0
                
                if 'HY' in df.columns and 'AY' in df.columns:
                    total_cards = row.get('HY', 0) + row.get('AY', 0)
                    features['over_4_5_cards'] = 1 if total_cards > 4.5 else 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _get_goals_scored(self, df, team):
        """Calculate average goals scored"""
        home_goals = df[df['HomeTeam'] == team]['FTHG'].mean()
        away_goals = df[df['AwayTeam'] == team]['FTAG'].mean()
        return np.nanmean([home_goals, away_goals])
    
    def _get_goals_conceded(self, df, team):
        """Calculate average goals conceded"""
        home_conc = df[df['HomeTeam'] == team]['FTAG'].mean()
        away_conc = df[df['AwayTeam'] == team]['FTHG'].mean()
        return np.nanmean([home_conc, away_conc])
    
    def _get_form(self, last_games, team):
        """Calculate form (win rate in last games)"""
        if len(last_games) == 0:
            return 0
        
        wins = 0
        for _, game in last_games.iterrows():
            if game['HomeTeam'] == team and game['FTR'] == 'H':
                wins += 1
            elif game['AwayTeam'] == team and game['FTR'] == 'A':
                wins += 1
        
        return wins / len(last_games)
    
    def _get_corners(self, df, team):
        """Calculate average corners"""
        home_corners = df[df['HomeTeam'] == team]['HC'].mean() if 'HC' in df.columns else 0
        away_corners = df[df['AwayTeam'] == team]['AC'].mean() if 'AC' in df.columns else 0
        return np.nanmean([home_corners, away_corners])
    
    def _get_cards(self, df, team):
        """Calculate average cards"""
        home_cards = df[df['HomeTeam'] == team]['HY'].mean() if 'HY' in df.columns else 0
        away_cards = df[df['AwayTeam'] == team]['AY'].mean() if 'AY' in df.columns else 0
        return np.nanmean([home_cards, away_cards])
    
    def train_models(self, df):
        """Train ML models"""
        features_df = self.prepare_features(df)
        
        if len(features_df) < 100:
            print("Not enough data to train models")
            return False
        
        # Drop rows with NaN in features or targets
        features_df = features_df.dropna()
        
        if len(features_df) < 50:
            print("Not enough valid data after cleaning")
            return False
        
        # Feature columns
        feature_cols = ['home_goals_avg', 'home_conceded_avg', 'home_form',
                       'away_goals_avg', 'away_conceded_avg', 'away_form',
                       'home_corners_avg', 'away_corners_avg',
                       'home_cards_avg', 'away_cards_avg']
        
        X = features_df[feature_cols]
        
        # Train Over/Under 2.5 model
        if 'over_2_5' in features_df.columns:
            y_over = features_df['over_2_5']
            self.over_under_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            self.over_under_model.fit(X, y_over)
            print("✓ Over/Under 2.5 model trained")
        
        # Train BTTS model
        if 'btts' in features_df.columns:
            y_btts = features_df['btts']
            self.btts_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.btts_model.fit(X, y_btts)
            print("✓ BTTS model trained")
        
        # Train Corners model
        if 'over_10_5_corners' in features_df.columns:
            y_corners = features_df['over_10_5_corners']
            self.corners_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            self.corners_model.fit(X, y_corners)
            print("✓ Corners model trained")
        
        # Train Cards model
        if 'over_4_5_cards' in features_df.columns:
            y_cards = features_df['over_4_5_cards']
            self.cards_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            self.cards_model.fit(X, y_cards)
            print("✓ Cards model trained")
        
        return True
    
    def monte_carlo_simulation(self, home_lambda, away_lambda, n_sims=10000):
        """Run Monte Carlo simulation"""
        np.random.seed(42)
        
        home_goals = np.random.poisson(home_lambda, n_sims)
        away_goals = np.random.poisson(away_lambda, n_sims)
        total_goals = home_goals + away_goals
        
        # Calculate probabilities
        over_2_5 = np.sum(total_goals > 2.5) / n_sims
        btts = np.sum((home_goals > 0) & (away_goals > 0)) / n_sims
        
        return {
            'over_2_5': over_2_5,
            'btts': btts,
            'home_goals_sim': home_goals,
            'away_goals_sim': away_goals
        }
    
    def predict_match(self, home_team, away_team, df, ml_weight=0.6):
        """Make hybrid ML + Monte Carlo prediction"""
        
        # Get team stats
        home_goals_avg = self._get_goals_scored(df, home_team)
        home_conceded_avg = self._get_goals_conceded(df, home_team)
        away_goals_avg = self._get_goals_scored(df, away_team)
        away_conceded_avg = self._get_goals_conceded(df, away_team)
        
        # Calculate expected goals
        exp_home = (home_goals_avg + away_conceded_avg) / 2
        exp_away = (away_goals_avg + home_conceded_avg) / 2
        
        # Monte Carlo simulation
        mc_results = self.monte_carlo_simulation(exp_home, exp_away)
        
        # ML prediction
        features = np.array([[
            home_goals_avg,
            home_conceded_avg,
            self._get_form(df[df['HomeTeam'] == home_team].tail(5), home_team),
            away_goals_avg,
            away_conceded_avg,
            self._get_form(df[df['AwayTeam'] == away_team].tail(5), away_team),
            self._get_corners(df, home_team),
            self._get_corners(df, away_team),
            self._get_cards(df, home_team),
            self._get_cards(df, away_team)
        ]])
        
        predictions = {}
        
        # Over/Under 2.5
        if self.over_under_model:
            ml_over = self.over_under_model.predict_proba(features)[0][1]
            mc_over = mc_results['over_2_5']
            hybrid_over = ml_weight * ml_over + (1 - ml_weight) * mc_over
            predictions['over_2_5'] = {
                'probability': hybrid_over,
                'ml_prob': ml_over,
                'mc_prob': mc_over
            }
        
        # BTTS
        if self.btts_model:
            ml_btts = self.btts_model.predict_proba(features)[0][1]
            mc_btts = mc_results['btts']
            hybrid_btts = ml_weight * ml_btts + (1 - ml_weight) * mc_btts
            predictions['btts'] = {
                'probability': hybrid_btts,
                'ml_prob': ml_btts,
                'mc_prob': mc_btts
            }
        
        # Expected goals
        predictions['expected_goals'] = {
            'home': exp_home,
            'away': exp_away,
            'total': exp_home + exp_away
        }
        
        # Asian Handicap
        predictions['asian_handicap'] = self._calculate_asian_handicap(
            mc_results['home_goals_sim'],
            mc_results['away_goals_sim']
        )
        
        # Goal Lines
        predictions['goal_lines'] = self._calculate_goal_lines(
            mc_results['home_goals_sim'],
            mc_results['away_goals_sim']
        )
        
        # Match result probabilities
        home_wins = np.sum(mc_results['home_goals_sim'] > mc_results['away_goals_sim']) / len(mc_results['home_goals_sim'])
        draws = np.sum(mc_results['home_goals_sim'] == mc_results['away_goals_sim']) / len(mc_results['home_goals_sim'])
        away_wins = 1 - home_wins - draws
        
        predictions['match_result'] = {
            'home_win': home_wins,
            'draw': draws,
            'away_win': away_wins
        }
        
        return predictions
    
    def _calculate_asian_handicap(self, home_goals, away_goals):
        """Calculate Asian Handicap from simulations"""
        n = len(home_goals)
        results = []
        
        for line in np.arange(-3.0, 3.5, 0.5):
            goal_diff = home_goals - away_goals
            
            if line < 0:
                home_win = np.sum(goal_diff > abs(line))
                push = np.sum(goal_diff == abs(line))
            elif line > 0:
                home_win = np.sum(goal_diff > -line)
                push = np.sum(goal_diff == -line)
            else:
                home_win = np.sum(goal_diff > 0)
                push = np.sum(goal_diff == 0)
            
            home_pct = (home_win + push * 0.5) / n * 100
            
            results.append({
                'line': line,
                'home': round(home_pct, 1),
                'away': round(100 - home_pct, 1)
            })
        
        return results
    
    def _calculate_goal_lines(self, home_goals, away_goals):
        """Calculate Over/Under goal lines"""
        total = home_goals + away_goals
        n = len(total)
        results = []
        
        for line in np.arange(0.5, 6.5, 0.5):
            over_pct = np.sum(total > line) / n * 100
            results.append({
                'line': line,
                'over': round(over_pct, 1),
                'under': round(100 - over_pct, 1)
            })
        
        return results

# Main execution
if __name__ == "__main__":
    print("⚽ Football Predictor - Command Line Mode")
    print("=" * 50)
    
    # League selection
    print("\nAvailable Leagues:")
    print("1. Premier League")
    print("2. La Liga")
    print("3. Serie A")
    print("4. Serie B")
    print("5. Bundesliga 2")
    
    league_choice = input("\nSelect league (1-5): ")
    
    league_map = {
        '1': ('E0', 'Premier League'),
        '2': ('SP1', 'La Liga'),
        '3': ('I1', 'Serie A'),
        '4': ('I2', 'Serie B'),
        '5': ('D2', 'Bundesliga 2')
    }
    
    if league_choice not in league_map:
        print("Invalid choice!")
        exit()
    
    league_code, league_name = league_map[league_choice]
    print(f"\n✓ Selected: {league_name}")
    
    # Load data
    print("\nLoading data...")
    predictor = FootballPredictor()
    seasons = ['2324', '2425', '2526']  # Last 3 seasons
    df = predictor.load_data(league_code, seasons)
    
    if df.empty:
        print("Failed to load data!")
        exit()
    
    print(f"\n✓ Loaded {len(df)} matches")
    
    # Train models
    print("\nTraining models...")
    if not predictor.train_models(df):
        print("Failed to train models!")
        exit()
    
    # Get teams
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    
    print(f"\n✓ {len(teams)} teams available")
    
    # Match prediction
    print("\n" + "=" * 50)
    print("MATCH PREDICTION")
    print("=" * 50)
    
    print("\nHome Team:")
    home_team = input("Enter team name: ")
    
    print("\nAway Team:")
    away_team = input("Enter team name: ")
    
    if home_team not in teams or away_team not in teams:
        print("\nError: Team not found in league!")
        exit()
    
    # Make prediction
    print("\nMaking prediction...")
    predictions = predictor.predict_match(home_team, away_team, df)
    
    # Display results
    print("\n" + "=" * 50)
    print(f"{home_team} vs {away_team}")
    print("=" * 50)
    
    print("\n🏆 MATCH RESULT:")
    print(f"Home Win: {predictions['match_result']['home_win']*100:.1f}%")
    print(f"Draw: {predictions['match_result']['draw']*100:.1f}%")
    print(f"Away Win: {predictions['match_result']['away_win']*100:.1f}%")
    
    print("\n⚽ EXPECTED GOALS:")
    print(f"{home_team}: {predictions['expected_goals']['home']:.2f}")
    print(f"{away_team}: {predictions['expected_goals']['away']:.2f}")
    print(f"Total: {predictions['expected_goals']['total']:.2f}")
    
    if 'over_2_5' in predictions:
        print("\n📊 OVER/UNDER 2.5 GOALS:")
        print(f"Over: {predictions['over_2_5']['probability']*100:.1f}%")
        print(f"Under: {(1-predictions['over_2_5']['probability'])*100:.1f}%")
    
    if 'btts' in predictions:
        print("\n🎯 BOTH TEAMS TO SCORE:")
        print(f"Yes: {predictions['btts']['probability']*100:.1f}%")
        print(f"No: {(1-predictions['btts']['probability'])*100:.1f}%")
    
    print("\n" + "=" * 50)
