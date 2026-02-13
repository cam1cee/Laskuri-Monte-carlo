import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Football Predictions", page_icon="⚽", layout="wide")

# League configurations - YOUR REQUESTED LEAGUES
LEAGUES = {
    'Premier League': 'E0',
    'La Liga': 'SP1',
    'Serie A': 'I1',
    'Serie B': 'I2',
    'Bundesliga 2': 'D2'
}

@st.cache_data(ttl=3600)
def load_league_data(code, season):
    """Load data from football-data.co.uk"""
    try:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
        df = pd.read_csv(url, encoding='latin1', on_bad_lines='skip')
        required = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        if all(col in df.columns for col in required):
            return df[required + [c for c in df.columns if c not in required]].dropna(subset=required)
    except:
        pass
    return None

@st.cache_data
def load_all_data(league_code):
    """Load all seasons for a league"""
    seasons = ['2324', '2425', '2526']
    data = []
    
    for season in seasons:
        df = load_league_data(league_code, season)
        if df is not None:
            df['Season'] = season
            data.append(df)
    
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

def get_team_stats(df, team):
    """Get team statistics"""
    home = df[df['HomeTeam'] == team]
    away = df[df['AwayTeam'] == team]
    
    return {
        'home_scored': home['FTHG'].mean() if len(home) > 0 else 0,
        'home_conceded': home['FTAG'].mean() if len(home) > 0 else 0,
        'away_scored': away['FTAG'].mean() if len(away) > 0 else 0,
        'away_conceded': away['FTHG'].mean() if len(away) > 0 else 0,
        'games': len(home) + len(away)
    }

def monte_carlo_predict(home_team, away_team, df, n_sims=10000):
    """Monte Carlo simulation"""
    
    home_stats = get_team_stats(df, home_team)
    away_stats = get_team_stats(df, away_team)
    
    if home_stats['games'] < 5 or away_stats['games'] < 5:
        return None
    
    # Expected goals
    exp_home = (home_stats['home_scored'] + away_stats['away_conceded']) / 2
    exp_away = (away_stats['away_scored'] + home_stats['home_conceded']) / 2
    
    exp_home = max(0.1, min(5.0, exp_home))
    exp_away = max(0.1, min(5.0, exp_away))
    
    # Run simulation
    np.random.seed(42)
    home_goals = np.random.poisson(exp_home, n_sims)
    away_goals = np.random.poisson(exp_away, n_sims)
    
    # Match results
    home_wins = np.sum(home_goals > away_goals) / n_sims
    draws = np.sum(home_goals == away_goals) / n_sims
    away_wins = np.sum(home_goals < away_goals) / n_sims
    
    # Over/Under & BTTS
    total_goals = home_goals + away_goals
    over_2_5 = np.sum(total_goals > 2.5) / n_sims
    btts = np.sum((home_goals > 0) & (away_goals > 0)) / n_sims
    
    return {
        'home_win': home_wins,
        'draw': draws,
        'away_win': away_wins,
        'exp_home': exp_home,
        'exp_away': exp_away,
        'over_2_5': over_2_5,
        'btts': btts,
        'home_goals_sim': home_goals,
        'away_goals_sim': away_goals
    }

def asian_handicap(home_goals, away_goals):
    """Calculate Asian Handicap"""
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

# Main UI
st.title("⚽ Football Betting Predictor")
st.markdown("### AI-powered predictions using Monte Carlo simulation")

with st.sidebar:
    st.header("Settings")
    
    league = st.selectbox("Select League", list(LEAGUES.keys()))
    league_code = LEAGUES[league]
    
    sims = st.selectbox("Simulations", [5000, 10000, 20000], index=1)
    
    if st.button("Load Data", type="primary", use_container_width=True):
        with st.spinner(f"Loading {league}..."):
            df = load_all_data(league_code)
            if not df.empty:
                st.session_state.df = df
                st.session_state.loaded = True
                st.success(f"✅ {len(df)} matches")
                st.rerun()
            else:
                st.error("Failed to load")

if 'loaded' not in st.session_state:
    st.session_state.loaded = False

if st.session_state.loaded:
    df = st.session_state.df
    
    # Get teams
    current = df[df['Season'] == '2526']
    if current.empty:
        current = df[df['Season'] == df['Season'].max()]
    
    teams = sorted(pd.concat([current['HomeTeam'], current['AwayTeam']]).unique())
    
    st.header("Match Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Home Team", teams)
    with col2:
        away = st.selectbox("Away Team", [t for t in teams if t != home])
    
    if st.button("Predict Match", type="primary", use_container_width=True):
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
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Over/Under 2.5")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Over", f"{pred['over_2_5']*100:.1f}%")
                with c2:
                    st.metric("Under", f"{(1-pred['over_2_5'])*100:.1f}%")
            
            with col2:
                st.subheader("🎯 Both Teams to Score")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Yes", f"{pred['btts']*100:.1f}%")
                with c2:
                    st.metric("No", f"{(1-pred['btts'])*100:.1f}%")
            
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
            st.error("Not enough data")

else:
    st.info("👈 Select league and click 'Load Data'")
    
    st.markdown("""
    ### Features:
    - **Monte Carlo Simulation** (5,000 - 20,000 simulations)
    - **Win Probabilities** (Home/Draw/Away)
    - **Expected Goals**
    - **Over/Under 2.5 Goals**
    - **Both Teams to Score**
    - **Asian Handicap** (-3.0 to +3.0)
    - **Goal Lines** (0.5 to 6.0)
    
    ### Leagues:
    🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League | 🇪🇸 La Liga | 🇮🇹 Serie A  
    🇮🇹 Serie B | 🇩🇪 2.Bundesliga
    
    ### Data: Seasons 2023-24, 2024-25, 2025-26
    """)

st.markdown("---")
st.markdown("*Data from [football-data.co.uk](https://www.football-data.co.uk/)*")
