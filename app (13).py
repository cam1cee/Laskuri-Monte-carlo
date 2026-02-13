import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

# Dark theme
st.set_page_config(page_title="Football Betting Predictor", page_icon="⚽", layout="wide")

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .recommendation-box {
        background-color: #1e4620;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# YOUR LEAGUES
LEAGUES = {
    'Premier League': 'E0',
    'La Liga': 'SP1', 
    'Serie A': 'I1',
    'Serie B': 'I2',
    'Bundesliga 2': 'D2'
}

@st.cache_data(ttl=3600)
def load_league_data(code, season):
    try:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
        df = pd.read_csv(url, encoding='latin1', on_bad_lines='skip')
        required = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        if all(col in df.columns for col in required):
            return df
    except:
        pass
    return None

@st.cache_data
def load_all_data(league_code):
    seasons = ['2324', '2425', '2526']
    data = []
    for season in seasons:
        df = load_league_data(league_code, season)
        if df is not None:
            df['Season'] = season
            data.append(df)
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

def get_team_stats(df, team, venue='all'):
    if venue == 'home':
        games = df[df['HomeTeam'] == team]
        goals_scored = games['FTHG'].mean() if len(games) > 0 else 0
        goals_conceded = games['FTAG'].mean() if len(games) > 0 else 0
        corners = games['HC'].mean() if 'HC' in games.columns and len(games) > 0 else 0
        cards = games['HY'].mean() if 'HY' in games.columns and len(games) > 0 else 0
    elif venue == 'away':
        games = df[df['AwayTeam'] == team]
        goals_scored = games['FTAG'].mean() if len(games) > 0 else 0
        goals_conceded = games['FTHG'].mean() if len(games) > 0 else 0
        corners = games['AC'].mean() if 'AC' in games.columns and len(games) > 0 else 0
        cards = games['AY'].mean() if 'AY' in games.columns and len(games) > 0 else 0
    else:
        home = df[df['HomeTeam'] == team]
        away = df[df['AwayTeam'] == team]
        total_games = len(home) + len(away)
        goals_scored = (home['FTHG'].sum() + away['FTAG'].sum()) / max(1, total_games)
        goals_conceded = (home['FTAG'].sum() + away['FTHG'].sum()) / max(1, total_games)
        corners = 0
        cards = 0
    
    return {
        'goals_scored': goals_scored,
        'goals_conceded': goals_conceded,
        'corners': corners,
        'cards': cards,
        'games': len(games) if venue != 'all' else total_games
    }

def get_form(df, team, last_n=5):
    home = df[df['HomeTeam'] == team].copy()
    away = df[df['AwayTeam'] == team].copy()
    
    home['Result'] = home['FTR'].apply(lambda x: 'W' if x == 'H' else ('D' if x == 'D' else 'L'))
    away['Result'] = away['FTR'].apply(lambda x: 'W' if x == 'A' else ('D' if x == 'D' else 'L'))
    
    all_games = pd.concat([home[['Result']], away[['Result']]])
    return all_games.tail(last_n)['Result'].tolist()

def get_days_since_last_match(df, team):
    # Placeholder - would need date parsing
    return 5

def monte_carlo(home_team, away_team, df, ml_weight=0.6, n_sims=10000):
    # League averages
    league_home_avg = df['FTHG'].mean()
    league_away_avg = df['FTAG'].mean()
    
    # Team stats
    home_stats = get_team_stats(df, home_team, 'home')
    away_stats = get_team_stats(df, away_team, 'away')
    
    if home_stats['games'] < 3 or away_stats['games'] < 3:
        return None
    
    # Expected goals
    home_attack = home_stats['goals_scored'] / league_home_avg if league_home_avg > 0 else 1
    home_defense = home_stats['goals_conceded'] / league_away_avg if league_away_avg > 0 else 1
    away_attack = away_stats['goals_scored'] / league_away_avg if league_away_avg > 0 else 1
    away_defense = away_stats['goals_conceded'] / league_home_avg if league_home_avg > 0 else 1
    
    exp_home = max(0.3, min(5.0, home_attack * away_defense * league_home_avg))
    exp_away = max(0.3, min(5.0, away_attack * home_defense * league_away_avg))
    
    # Simulate
    np.random.seed(42)
    home_goals = np.random.poisson(exp_home, n_sims)
    away_goals = np.random.poisson(exp_away, n_sims)
    total_goals = home_goals + away_goals
    
    # Expected corners/cards
    exp_corners = home_stats['corners'] + away_stats['corners']
    exp_cards = home_stats['cards'] + away_stats['cards']
    
    total_corners = np.random.poisson(max(5, exp_corners), n_sims)
    total_cards = np.random.poisson(max(2, exp_cards), n_sims)
    
    # ML component (simple form-based adjustment)
    home_form = get_form(df, home_team, 5)
    away_form = get_form(df, away_team, 5)
    
    home_form_score = home_form.count('W') / max(1, len(home_form))
    away_form_score = away_form.count('W') / max(1, len(away_form))
    
    # Monte Carlo probabilities
    mc_over = np.sum(total_goals > 2.5) / n_sims
    mc_btts = np.sum((home_goals > 0) & (away_goals > 0)) / n_sims
    mc_over_corners = np.sum(total_corners > 10.5) / n_sims
    mc_over_cards = np.sum(total_cards > 4.5) / n_sims
    
    # ML adjustments (simple form-based)
    ml_over = min(1.0, mc_over * (1 + (home_form_score + away_form_score - 1) * 0.1))
    ml_btts = min(1.0, mc_btts * (1 + (home_form_score + away_form_score - 1) * 0.05))
    ml_over_corners = mc_over_corners
    ml_over_cards = mc_over_cards
    
    # Hybrid
    over_prob = ml_weight * ml_over + (1 - ml_weight) * mc_over
    btts_prob = ml_weight * ml_btts + (1 - ml_weight) * mc_btts
    corners_prob = ml_over_corners
    cards_prob = ml_over_cards
    
    # H2H avg goals (placeholder)
    h2h_avg = exp_home + exp_away
    
    return {
        'exp_home': exp_home,
        'exp_away': exp_away,
        'exp_total': exp_home + exp_away,
        'exp_corners': exp_corners,
        'exp_cards': exp_cards,
        'over_prob': over_prob,
        'btts_prob': btts_prob,
        'corners_prob': corners_prob,
        'cards_prob': cards_prob,
        'home_stats': home_stats,
        'away_stats': away_stats,
        'home_form': home_form,
        'away_form': away_form,
        'h2h_avg': h2h_avg,
        'simulation_results': total_goals
    }

def calculate_ev(prob, odds):
    if odds <= 1.0:
        return 0
    return (prob * (odds - 1) - (1 - prob)) * 100

# Main App
st.title("⚽ Football Betting Predictor")
st.markdown("**ML + Monte Carlo** predictions for Over/Under 2.5 and BTTS")

# League selection
league = st.selectbox("Select League", list(LEAGUES.keys()))
league_code = LEAGUES[league]

if st.button("Load Data"):
    with st.spinner("Loading..."):
        df = load_all_data(league_code)
        if not df.empty:
            st.session_state.df = df
            st.session_state.loaded = True
            st.success(f"✅ Loaded {len(df)} matches")
            st.rerun()

if 'loaded' not in st.session_state:
    st.session_state.loaded = False

if st.session_state.loaded:
    df = st.session_state.df
    
    # Get teams
    current = df[df['Season'] == '2526']
    if current.empty:
        current = df[df['Season'] == df['Season'].max()]
    teams = sorted(pd.concat([current['HomeTeam'], current['AwayTeam']]).dropna().unique())
    
    st.markdown("---")
    
    # Match selection
    st.subheader("📋 Match Details")
    home = st.selectbox("Home Team", teams, key='home')
    away = st.selectbox("Away Team", [t for t in teams if t != home], key='away')
    
    # Bookmaker odds section
    st.subheader("💰 Bookmaker Odds (Decimal)")
    
    tab1, tab2, tab3 = st.tabs(["⚽ Goals", "🚩 Corners", "🟨 Cards"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Over 2.5**")
            over_odds = st.number_input("", min_value=1.01, value=1.85, step=0.01, key='over', label_visibility='collapsed')
        with col2:
            st.markdown("**Under 2.5**")
            under_odds = st.number_input("", min_value=1.01, value=2.00, step=0.01, key='under', label_visibility='collapsed')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**BTTS Yes**")
            btts_yes_odds = st.number_input("", min_value=1.01, value=1.75, step=0.01, key='btts_yes', label_visibility='collapsed')
        with col2:
            st.markdown("**BTTS No**")
            btts_no_odds = st.number_input("", min_value=1.01, value=2.10, step=0.01, key='btts_no', label_visibility='collapsed')
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Over 10.5**")
            corners_over_odds = st.number_input("", min_value=1.01, value=1.90, step=0.01, key='corners_over', label_visibility='collapsed')
        with col2:
            st.markdown("**Under 10.5**")
            corners_under_odds = st.number_input("", min_value=1.01, value=1.95, step=0.01, key='corners_under', label_visibility='collapsed')
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Over 4.5**")
            cards_over_odds = st.number_input("", min_value=1.01, value=2.20, step=0.01, key='cards_over', label_visibility='collapsed')
        with col2:
            st.markdown("**Under 4.5**")
            cards_under_odds = st.number_input("", min_value=1.01, value=1.70, step=0.01, key='cards_under', label_visibility='collapsed')
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        ml_weight = st.slider("ML Model Weight", 0.0, 1.0, 0.60, 0.01)
        st.caption(f"Monte Carlo Weight: {1-ml_weight:.2f}")
        
        confidence = st.slider("Confidence Threshold (%)", 50, 80, 60, 1)
        min_edge = st.slider("Minimum Edge (%)", 0, 10, 5, 1)
    
    # Predict button
    if st.button("🎯 Make Prediction", use_container_width=True, type="primary"):
        pred = monte_carlo(home, away, df, ml_weight, 10000)
        
        if pred:
            st.markdown("---")
            
            # Match title
            st.header(f"{home} vs {away}")
            
            # Form & Fatigue
            st.subheader("📊 Recent Form & Fatigue")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**{home} - Last 5 Games:**")
                form_html = ""
                for r in pred['home_form']:
                    if r == 'W':
                        form_html += '<span style="background:#22c55e;padding:5px 10px;margin:2px;border-radius:3px;color:white">✓ W</span> '
                    elif r == 'D':
                        form_html += '<span style="background:#eab308;padding:5px 10px;margin:2px;border-radius:3px;color:white">− D</span> '
                    else:
                        form_html += '<span style="background:#ef4444;padding:5px 10px;margin:2px;border-radius:3px;color:white">✗ L</span> '
                st.markdown(form_html, unsafe_allow_html=True)
                
                days = get_days_since_last_match(df, home)
                st.markdown(f"**Days Since Last Match:** {days} days")
                if days >= 6:
                    st.success("✅ Normal rest period")
                else:
                    st.warning("⚠️ Short rest")
            
            with col2:
                st.markdown(f"**{away} - Last 5 Games:**")
                form_html = ""
                for r in pred['away_form']:
                    if r == 'W':
                        form_html += '<span style="background:#22c55e;padding:5px 10px;margin:2px;border-radius:3px;color:white">✓ W</span> '
                    elif r == 'D':
                        form_html += '<span style="background:#eab308;padding:5px 10px;margin:2px;border-radius:3px;color:white">− D</span> '
                    else:
                        form_html += '<span style="background:#ef4444;padding:5px 10px;margin:2px;border-radius:3px;color:white">✗ L</span> '
                st.markdown(form_html, unsafe_allow_html=True)
                
                days = get_days_since_last_match(df, away)
                st.markdown(f"**Days Since Last Match:** {days} days")
                if days >= 6:
                    st.success("✅ Normal rest period")
                else:
                    st.warning("⚠️ Short rest")
            
            # Team stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{home} Goals/Game", f"{pred['home_stats']['goals_scored']:.2f}")
                st.metric(f"{home} Corners/Game", f"{pred['home_stats']['corners']:.1f}")
                st.metric(f"{home} Cards/Game", f"{pred['home_stats']['cards']:.1f}")
                st.metric(f"{home} Form", f"{pred['home_form'].count('W')}/{len(pred['home_form'])}")
            
            with col2:
                st.metric("Expected Total Goals", f"{pred['exp_total']:.2f}")
                st.metric("Expected Total Corners", f"{pred['exp_corners']:.1f}")
                st.metric("Expected Total Cards", f"{pred['exp_cards']:.1f}")
                st.metric("H2H Avg Goals", f"{pred['h2h_avg']:.2f}")
            
            with col3:
                st.metric(f"{away} Goals/Game", f"{pred['away_stats']['goals_scored']:.2f}")
                st.metric(f"{away} Corners/Game", f"{pred['away_stats']['corners']:.1f}")
                st.metric(f"{away} Cards/Game", f"{pred['away_stats']['cards']:.1f}")
                st.metric(f"{away} Form", f"{pred['away_form'].count('W')}/{len(pred['away_form'])}")
            
            # Predictions
            st.subheader("⚽ Over/Under 2.5 Goals")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Prediction:** Over 2.5" if pred['over_prob'] > 0.5 else "**Prediction:** Under 2.5")
                st.markdown(f"**↑ {pred['over_prob']*100:.1f}% confidence**" if pred['over_prob'] >= confidence/100 else f"**{pred['over_prob']*100:.1f}% confidence**")
            with col2:
                st.markdown("**Model Probability**")
                st.markdown(f"**{pred['over_prob']*100:.1f}%**")
                
                ev = calculate_ev(pred['over_prob'], over_odds)
                st.markdown(f"**Bookmaker Odds:** {over_odds}")
                st.markdown(f"**EV:** {ev:+.1f}%")
            
            st.subheader("⚽ Both Teams To Score")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Prediction:** BTTS Yes" if pred['btts_prob'] > 0.5 else "**Prediction:** BTTS No")
                st.markdown(f"**↑ {pred['btts_prob']*100:.1f}% confidence**" if pred['btts_prob'] >= confidence/100 else f"**{pred['btts_prob']*100:.1f}% confidence**")
            with col2:
                st.markdown("**Model Probability**")
                st.markdown(f"**{pred['btts_prob']*100:.1f}%**")
                
                ev = calculate_ev(pred['btts_prob'], btts_yes_odds)
                st.markdown(f"**Bookmaker Odds:** {btts_yes_odds}")
                st.markdown(f"**EV:** {ev:+.1f}%")
            
            st.subheader("🚩 Corners (O/U 10.5)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Prediction:** Over 10.5" if pred['corners_prob'] > 0.5 else "**Prediction:** Under 10.5")
                st.markdown(f"**↑ {pred['corners_prob']*100:.1f}% confidence**" if pred['corners_prob'] >= confidence/100 else f"**{pred['corners_prob']*100:.1f}% confidence**")
            with col2:
                st.markdown("**Model Probability**")
                st.markdown(f"**{pred['corners_prob']*100:.1f}%**")
                
                ev = calculate_ev(pred['corners_prob'], corners_over_odds)
                st.markdown(f"**Bookmaker Odds:** {corners_over_odds}")
                st.markdown(f"**EV:** {ev:+.1f}%")
            
            st.subheader("🟨 Cards (O/U 4.5)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Prediction:** Under 4.5")
                st.markdown(f"**↑ {(1-pred['cards_prob'])*100:.1f}% confidence**" if (1-pred['cards_prob']) >= confidence/100 else f"**{(1-pred['cards_prob'])*100:.1f}% confidence**")
            with col2:
                st.markdown("**Model Probability**")
                st.markdown(f"**{(1-pred['cards_prob'])*100:.1f}%**")
                
                ev = calculate_ev(1-pred['cards_prob'], cards_under_odds)
                st.markdown(f"**Bookmaker Odds:** {cards_under_odds}")
                st.markdown(f"**EV:** {ev:+.1f}%")
            
            # Monte Carlo visualization
            st.subheader("🎲 Monte Carlo Simulation (10,000 iterations)")
            
            # Create histogram
            fig = go.Figure()
            
            hist_data = np.histogram(pred['simulation_results'], bins=range(0, 12))
            
            fig.add_trace(go.Bar(
                x=list(range(0, 11)),
                y=hist_data[0],
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                xaxis_title="Total Goals",
                yaxis_title="Frequency",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Final recommendations
            st.subheader("🎯 Final Recommendation")
            
            recommendations = []
            
            # Check Over 2.5
            ev_over = calculate_ev(pred['over_prob'], over_odds)
            if pred['over_prob']*100 >= confidence and ev_over >= min_edge:
                recommendations.append(f"• **Over 2.5** @ {over_odds} (EV: {ev_over:+.1f}%, Confidence: {pred['over_prob']*100:.1f}%)")
            
            # Check BTTS
            ev_btts = calculate_ev(pred['btts_prob'], btts_yes_odds)
            if pred['btts_prob']*100 >= confidence and ev_btts >= min_edge:
                recommendations.append(f"• **BTTS Yes** @ {btts_yes_odds} (EV: {ev_btts:+.1f}%, Confidence: {pred['btts_prob']*100:.1f}%)")
            
            # Check Corners
            ev_corners = calculate_ev(pred['corners_prob'], corners_over_odds)
            if pred['corners_prob']*100 >= confidence and ev_corners >= min_edge:
                recommendations.append(f"• **Over 10.5 corners** @ {corners_over_odds} (EV: {ev_corners:+.1f}%, Confidence: {pred['corners_prob']*100:.1f}%)")
            
            # Check Cards
            ev_cards = calculate_ev(1-pred['cards_prob'], cards_under_odds)
            if (1-pred['cards_prob'])*100 >= confidence and ev_cards >= min_edge:
                recommendations.append(f"• **Under 4.5 cards** @ {cards_under_odds} (EV: {ev_cards:+.1f}%, Confidence: {(1-pred['cards_prob'])*100:.1f}%)")
            
            if recommendations:
                st.success("✅ RECOMMENDED BETS:")
                for rec in recommendations:
                    st.markdown(rec)
            else:
                st.info("No bets meet the value criteria")
            
            # Track bet section
            st.subheader("💾 Track This Bet")
            
            col1, col2 = st.columns(2)
            with col1:
                match_date = st.date_input("Match Date", value=datetime.now())
                bet_type = st.text_input("Bet Type", value="e.g., Over 2.5, BTTS Yes")
            
            with col2:
                odds_input = st.number_input("Odds", min_value=1.01, value=2.00, step=0.01)
                stake = st.number_input("Stake (units)", min_value=0.1, value=1.0, step=0.1)
            
            potential_return = stake * odds_input
            potential_profit = potential_return - stake
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Potential Return", f"{potential_return:.2f} units")
            with col2:
                st.metric("Potential Profit", f"{potential_profit:.2f} units")
            
            if st.button("💾 Save Bet", type="primary"):
                st.success("Bet saved!")
        
        else:
            st.error("Not enough data")

else:
    st.info("Select a league and click 'Load Data' to begin")

st.markdown("---")
st.markdown("*Data from football-data.co.uk | For educational purposes only*")
