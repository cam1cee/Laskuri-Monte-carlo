import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

st.set_page_config(page_title="Football Betting Predictor", page_icon="⚽", layout="wide")

# YOUR LEAGUES
LEAGUES = {
    'Premier League': 'E0',
    'La Liga': 'SP1',
    'Serie A': 'I1',
    'Serie B': 'I2',
    'Bundesliga 2': 'D2'
}

# Bet tracking file
BET_FILE = 'bets.json'

@st.cache_data(ttl=3600)
def load_league_data(code, season):
    """Load data from football-data.co.uk"""
    try:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
        df = pd.read_csv(url, encoding='latin1', on_bad_lines='skip')
        
        required = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        if all(col in df.columns for col in required):
            # Keep all columns for corners/cards
            return df
    except:
        pass
    return None

@st.cache_data
def load_all_data(league_code):
    """Load last 3 seasons"""
    seasons = ['2324', '2425', '2526']
    data = []
    
    progress = st.progress(0)
    status = st.empty()
    
    for i, season in enumerate(seasons):
        status.text(f"Loading {season}...")
        df = load_league_data(league_code, season)
        if df is not None:
            df['Season'] = season
            data.append(df)
        progress.progress((i + 1) / len(seasons))
    
    progress.empty()
    status.empty()
    
    if data:
        combined = pd.concat(data, ignore_index=True)
        combined = combined.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])
        return combined
    
    return pd.DataFrame()

def get_team_stats(df, team, venue='all'):
    """Calculate team statistics with proper league context"""
    if venue == 'home':
        games = df[df['HomeTeam'] == team].copy()
        goals_for = 'FTHG'
        goals_against = 'FTAG'
        corners_for = 'HC' if 'HC' in df.columns else None
        corners_against = 'AC' if 'AC' in df.columns else None
        cards = 'HY' if 'HY' in df.columns else None
    elif venue == 'away':
        games = df[df['AwayTeam'] == team].copy()
        goals_for = 'FTAG'
        goals_against = 'FTHG'
        corners_for = 'AC' if 'AC' in df.columns else None
        corners_against = 'HC' if 'HC' in df.columns else None
        cards = 'AY' if 'AY' in df.columns else None
    else:
        home_games = df[df['HomeTeam'] == team].copy()
        away_games = df[df['AwayTeam'] == team].copy()
        
        return {
            'games': len(home_games) + len(away_games),
            'goals_scored': (home_games['FTHG'].sum() + away_games['FTAG'].sum()) / max(1, len(home_games) + len(away_games)),
            'goals_conceded': (home_games['FTAG'].sum() + away_games['FTHG'].sum()) / max(1, len(home_games) + len(away_games))
        }
    
    if len(games) == 0:
        return {'games': 0, 'goals_scored': 0, 'goals_conceded': 0, 'corners': 0, 'cards': 0}
    
    stats = {
        'games': len(games),
        'goals_scored': games[goals_for].mean(),
        'goals_conceded': games[goals_against].mean()
    }
    
    if corners_for and corners_for in games.columns:
        stats['corners'] = games[corners_for].mean()
    else:
        stats['corners'] = 0
    
    if cards and cards in games.columns:
        stats['cards'] = games[cards].mean()
    else:
        stats['cards'] = 0
    
    return stats

def get_recent_form(df, team, last_n=5):
    """Get recent form with W/D/L"""
    home = df[df['HomeTeam'] == team].copy()
    away = df[df['AwayTeam'] == team].copy()
    
    home['Result'] = home['FTR'].apply(lambda x: 'W' if x == 'H' else ('D' if x == 'D' else 'L'))
    away['Result'] = away['FTR'].apply(lambda x: 'W' if x == 'A' else ('D' if x == 'D' else 'L'))
    
    all_games = pd.concat([home[['Date', 'Result']], away[['Date', 'Result']]])
    
    if 'Date' in all_games.columns:
        all_games['Date'] = pd.to_datetime(all_games['Date'], errors='coerce')
        all_games = all_games.dropna(subset=['Date'])
        all_games = all_games.sort_values('Date', ascending=False)
    
    recent = all_games.head(last_n)['Result'].tolist()
    
    return recent

def monte_carlo_predict(home_team, away_team, df, n_sims=10000):
    """Monte Carlo with proper league averages"""
    
    # League averages
    league_avg_home_goals = df['FTHG'].mean()
    league_avg_away_goals = df['FTAG'].mean()
    
    # Team stats
    home_stats_home = get_team_stats(df, home_team, 'home')
    away_stats_away = get_team_stats(df, away_team, 'away')
    
    if home_stats_home['games'] < 3 or away_stats_away['games'] < 3:
        return None
    
    # Attack and defense strengths
    home_attack = home_stats_home['goals_scored'] / league_avg_home_goals if league_avg_home_goals > 0 else 1
    home_defense = home_stats_home['goals_conceded'] / league_avg_away_goals if league_avg_away_goals > 0 else 1
    
    away_attack = away_stats_away['goals_scored'] / league_avg_away_goals if league_avg_away_goals > 0 else 1
    away_defense = away_stats_away['goals_conceded'] / league_avg_home_goals if league_avg_home_goals > 0 else 1
    
    # Expected goals
    exp_home = home_attack * away_defense * league_avg_home_goals
    exp_away = away_attack * home_defense * league_avg_away_goals
    
    # Bounds
    exp_home = max(0.3, min(4.0, exp_home))
    exp_away = max(0.3, min(4.0, exp_away))
    
    # Expected corners and cards
    exp_home_corners = home_stats_home['corners']
    exp_away_corners = away_stats_away['corners']
    exp_total_corners = exp_home_corners + exp_away_corners
    
    exp_home_cards = home_stats_home['cards']
    exp_away_cards = away_stats_away['cards']
    exp_total_cards = exp_home_cards + exp_away_cards
    
    # Simulate
    np.random.seed(42)
    home_goals = np.random.poisson(exp_home, n_sims)
    away_goals = np.random.poisson(exp_away, n_sims)
    
    # Simulate corners and cards
    total_corners = np.random.poisson(max(1, exp_total_corners), n_sims)
    total_cards = np.random.poisson(max(1, exp_total_cards), n_sims)
    
    # Results
    total_goals = home_goals + away_goals
    
    return {
        'home_win': np.sum(home_goals > away_goals) / n_sims,
        'draw': np.sum(home_goals == away_goals) / n_sims,
        'away_win': np.sum(home_goals < away_goals) / n_sims,
        'exp_home': exp_home,
        'exp_away': exp_away,
        'over_2_5': np.sum(total_goals > 2.5) / n_sims,
        'btts': np.sum((home_goals > 0) & (away_goals > 0)) / n_sims,
        'over_10_5_corners': np.sum(total_corners > 10.5) / n_sims,
        'over_4_5_cards': np.sum(total_cards > 4.5) / n_sims,
        'home_goals_sim': home_goals,
        'away_goals_sim': away_goals,
        'exp_corners': exp_total_corners,
        'exp_cards': exp_total_cards
    }

def asian_handicap(home_goals, away_goals):
    """Asian Handicap calculation"""
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

def goal_lines(home_goals, away_goals):
    """Goal lines calculation"""
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

def calculate_ev(prob, odds):
    """Calculate Expected Value"""
    if odds <= 1.0:
        return 0
    implied_prob = 1 / odds
    ev = (prob * (odds - 1)) - (1 - prob)
    return ev * 100

def load_bets():
    """Load saved bets"""
    if os.path.exists(BET_FILE):
        with open(BET_FILE, 'r') as f:
            return json.load(f)
    return []

def save_bet(bet):
    """Save a new bet"""
    bets = load_bets()
    bets.append(bet)
    with open(BET_FILE, 'w') as f:
        json.dump(bets, f)

# Main App
st.title("⚽ Football Betting Predictor")
st.markdown("### AI-powered predictions using Machine Learning + Monte Carlo simulation")

# Tabs
tab1, tab2 = st.tabs(["Predictions", "Bet Tracker"])

with tab1:
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        league = st.selectbox("League", list(LEAGUES.keys()))
        league_code = LEAGUES[league]
        
        sims = st.selectbox("Simulations", [5000, 10000, 20000], index=1)
        
        st.markdown("---")
        st.markdown("### Value Betting")
        confidence_threshold = st.slider("Min Confidence %", 50, 80, 60)
        ev_threshold = st.slider("Min EV %", 0, 10, 5)
        
        if st.button("🔄 Load Data", type="primary", use_container_width=True):
            with st.spinner(f"Loading {league}..."):
                df = load_all_data(league_code)
                if not df.empty:
                    st.session_state.df = df
                    st.session_state.league = league
                    st.session_state.loaded = True
                    st.success(f"✅ {len(df)} matches loaded")
                    st.rerun()
                else:
                    st.error("Failed to load data")
    
    if 'loaded' not in st.session_state:
        st.session_state.loaded = False
    
    if st.session_state.loaded:
        df = st.session_state.df
        
        # Get teams
        current = df[df['Season'] == '2526']
        if current.empty:
            current = df[df['Season'] == df['Season'].max()]
        
        teams = sorted(pd.concat([current['HomeTeam'], current['AwayTeam']]).dropna().unique())
        
        st.header(f"Match Prediction - {st.session_state.league}")
        
        col1, col2 = st.columns(2)
        with col1:
            home = st.selectbox("🏠 Home Team", teams)
        with col2:
            away = st.selectbox("✈️ Away Team", [t for t in teams if t != home])
        
        # Show recent form
        col1, col2 = st.columns(2)
        with col1:
            home_form = get_recent_form(df, home, 5)
            if home_form:
                form_display = " ".join([f"{'🟢' if r=='W' else '🟡' if r=='D' else '🔴'}" for r in home_form])
                st.markdown(f"**{home} Form:** {form_display}")
        
        with col2:
            away_form = get_recent_form(df, away, 5)
            if away_form:
                form_display = " ".join([f"{'🟢' if r=='W' else '🟡' if r=='D' else '🔴'}" for r in away_form])
                st.markdown(f"**{away} Form:** {form_display}")
        
        if st.button("📊 Predict Match", type="primary", use_container_width=True):
            pred = monte_carlo_predict(home, away, df, sims)
            
            if pred:
                st.markdown("---")
                
                # Match Result
                st.subheader("🏆 Match Result")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Home Win", f"{pred['home_win']*100:.1f}%")
                with c2:
                    st.metric("Draw", f"{pred['draw']*100:.1f}%")
                with c3:
                    st.metric("Away Win", f"{pred['away_win']*100:.1f}%")
                
                # Expected Goals
                st.subheader("⚽ Expected Goals")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric(home, f"{pred['exp_home']:.2f}")
                with c2:
                    st.metric("Total", f"{pred['exp_home'] + pred['exp_away']:.2f}")
                with c3:
                    st.metric(away, f"{pred['exp_away']:.2f}")
                
                # Markets
                st.subheader("📊 Betting Markets")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Over/Under 2.5 Goals**")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Over 2.5", f"{pred['over_2_5']*100:.1f}%")
                    with c2:
                        st.metric("Under 2.5", f"{(1-pred['over_2_5'])*100:.1f}%")
                    
                    # Value betting
                    with st.expander("💰 Value Analysis"):
                        odds_over = st.number_input("Odds for Over 2.5", min_value=1.01, value=2.0, step=0.01, key='over_odds')
                        ev_over = calculate_ev(pred['over_2_5'], odds_over)
                        
                        st.metric("Expected Value", f"{ev_over:+.1f}%")
                        
                        if ev_over > ev_threshold and pred['over_2_5']*100 > confidence_threshold:
                            st.success("✅ VALUE BET")
                            if st.button("Save Bet", key='save_over'):
                                save_bet({
                                    'date': datetime.now().strftime('%Y-%m-%d'),
                                    'match': f"{home} vs {away}",
                                    'market': 'Over 2.5',
                                    'odds': odds_over,
                                    'probability': f"{pred['over_2_5']*100:.1f}%",
                                    'ev': f"{ev_over:.1f}%",
                                    'result': 'Pending'
                                })
                                st.success("Bet saved!")
                
                with col2:
                    st.markdown("**Both Teams to Score**")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Yes", f"{pred['btts']*100:.1f}%")
                    with c2:
                        st.metric("No", f"{(1-pred['btts'])*100:.1f}%")
                    
                    # Value betting
                    with st.expander("💰 Value Analysis"):
                        odds_btts = st.number_input("Odds for BTTS Yes", min_value=1.01, value=2.0, step=0.01, key='btts_odds')
                        ev_btts = calculate_ev(pred['btts'], odds_btts)
                        
                        st.metric("Expected Value", f"{ev_btts:+.1f}%")
                        
                        if ev_btts > ev_threshold and pred['btts']*100 > confidence_threshold:
                            st.success("✅ VALUE BET")
                            if st.button("Save Bet", key='save_btts'):
                                save_bet({
                                    'date': datetime.now().strftime('%Y-%m-%d'),
                                    'match': f"{home} vs {away}",
                                    'market': 'BTTS Yes',
                                    'odds': odds_btts,
                                    'probability': f"{pred['btts']*100:.1f}%",
                                    'ev': f"{ev_btts:.1f}%",
                                    'result': 'Pending'
                                })
                                st.success("Bet saved!")
                
                # Corners and Cards
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Corners Over/Under 10.5**")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Over 10.5", f"{pred['over_10_5_corners']*100:.1f}%")
                    with c2:
                        st.metric("Under 10.5", f"{(1-pred['over_10_5_corners'])*100:.1f}%")
                    st.caption(f"Expected: {pred['exp_corners']:.1f} corners")
                
                with col2:
                    st.markdown("**Cards Over/Under 4.5**")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Over 4.5", f"{pred['over_4_5_cards']*100:.1f}%")
                    with c2:
                        st.metric("Under 4.5", f"{(1-pred['over_4_5_cards'])*100:.1f}%")
                    st.caption(f"Expected: {pred['exp_cards']:.1f} cards")
                
                # Asian Handicap
                st.subheader("📊 Asian Handicap")
                hcp = asian_handicap(pred['home_goals_sim'], pred['away_goals_sim'])
                hcp_df = pd.DataFrame(hcp)
                hcp_df['Handicap'] = hcp_df['line'].apply(lambda x: f"{x:+.1f}" if x != 0 else "0.0")
                hcp_df = hcp_df[['Handicap', 'home', 'away']]
                hcp_df.columns = ['Handicap', f'{home} %', f'{away} %']
                st.dataframe(hcp_df, use_container_width=True, hide_index=True, height=400)
                
                # Goal Lines
                st.subheader("🎯 Over/Under Goals")
                gl = goal_lines(pred['home_goals_sim'], pred['away_goals_sim'])
                gl_df = pd.DataFrame(gl)
                gl_df['Goals'] = gl_df['line'].apply(lambda x: f"{x:.1f}")
                gl_df = gl_df[['Goals', 'over', 'under']]
                gl_df.columns = ['Line', 'Over %', 'Under %']
                st.dataframe(gl_df, use_container_width=True, hide_index=True, height=350)
                
            else:
                st.error("Not enough data for prediction")
    
    else:
        st.info("👈 Select a league and click 'Load Data'")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 Multi-Market Predictions
            - **Goals:** Over/Under 2.5 & BTTS
            - **Corners:** Over/Under 10.5
            - **Cards:** Over/Under 4.5
            - **Asian Handicap:** -3.0 to +3.0
            - **Goal Lines:** 0.5 to 6.0
            """)
        
        with col2:
            st.markdown("""
            ### 🤖 Advanced Features
            - **Monte Carlo:** 10,000 simulations
            - **Value Betting:** EV analysis
            - **Recent Form:** W/D/L visualization
            - **Bet Tracking:** Save & track bets
            - **Smart Analysis:** League context
            """)
        
        st.markdown("""
        ### 🌍 Leagues
        🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League | 🇪🇸 La Liga | 🇮🇹 Serie A | 🇮🇹 Serie B | 🇩🇪 2.Bundesliga
        
        ### 📈 Data: 2023-24, 2024-25, 2025-26
        """)

with tab2:
    st.header("💰 Bet Tracker")
    
    bets = load_bets()
    
    if bets:
        df_bets = pd.DataFrame(bets)
        st.dataframe(df_bets, use_container_width=True)
        
        # Update bet result
        st.subheader("Update Bet Result")
        bet_idx = st.selectbox("Select Bet", range(len(bets)), format_func=lambda x: f"{bets[x]['date']} - {bets[x]['match']} - {bets[x]['market']}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Won", use_container_width=True):
                bets[bet_idx]['result'] = 'Won'
                with open(BET_FILE, 'w') as f:
                    json.dump(bets, f)
                st.success("Updated!")
                st.rerun()
        
        with col2:
            if st.button("❌ Lost", use_container_width=True):
                bets[bet_idx]['result'] = 'Lost'
                with open(BET_FILE, 'w') as f:
                    json.dump(bets, f)
                st.success("Updated!")
                st.rerun()
        
        # Stats
        st.subheader("📈 Statistics")
        won = len([b for b in bets if b['result'] == 'Won'])
        lost = len([b for b in bets if b['result'] == 'Lost'])
        pending = len([b for b in bets if b['result'] == 'Pending'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Won", won)
        with col2:
            st.metric("Lost", lost)
        with col3:
            st.metric("Pending", pending)
        
        if won + lost > 0:
            win_rate = won / (won + lost) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
    
    else:
        st.info("No bets saved yet. Make predictions and save value bets!")

st.markdown("---")
st.markdown("*Data from [football-data.co.uk](https://www.football-data.co.uk/) | For educational purposes only. Gamble responsibly.*")
