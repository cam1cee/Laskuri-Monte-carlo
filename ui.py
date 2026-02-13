import streamlit as st
import pandas as pd
import numpy as np
from predictor import FootballPredictor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Football Predictions",
    page_icon="⚽",
    layout="wide"
)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = FootballPredictor()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Title
st.title("⚽ Football Betting Predictor")
st.markdown("### AI-powered predictions using Machine Learning + Monte Carlo simulation")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # League selection
    league_options = {
        'Premier League': 'E0',
        'La Liga': 'SP1',
        'Serie A': 'I1',
        'Serie B': 'I2',
        'Bundesliga 2': 'D2'
    }
    
    selected_league = st.selectbox(
        "Select League",
        list(league_options.keys())
    )
    
    league_code = league_options[selected_league]
    
    # ML/MC Weight
    ml_weight = st.slider(
        "ML/MC Weight (ML%)",
        0, 100, 60
    ) / 100
    
    # Load data button
    if st.button("Load Data & Train", type="primary", use_container_width=True):
        with st.spinner(f"Loading {selected_league} data..."):
            seasons = ['2324', '2425', '2526']
            df = st.session_state.predictor.load_data(league_code, seasons)
            
            if not df.empty:
                st.session_state.df = df
                
                # Train models
                with st.spinner("Training models..."):
                    success = st.session_state.predictor.train_models(df)
                    
                    if success:
                        st.session_state.data_loaded = True
                        st.success(f"✅ Loaded {len(df)} matches")
                        st.rerun()
                    else:
                        st.error("Failed to train models")
            else:
                st.error("Failed to load data")

# Main content
if not st.session_state.data_loaded:
    st.info("👈 Select a league and click 'Load Data & Train' to begin")
    
    # Show features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Multi-Market Predictions
        - **Goals:** Over/Under 2.5 & BTTS
        - **Asian Handicap:** Full coverage
        - **Goal Lines:** 0.5 to 6.0
        - **Match Result:** Home/Draw/Away
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Dual System
        - **Machine Learning:** GB + RF models
        - **Monte Carlo:** 10,000 simulations
        - **Hybrid:** Weighted combination
        - **Smart Analysis:** Form & stats
        """)
    
    st.markdown("""
    ### 🌍 Supported Leagues
    - 🏴󠁧󠁢󠁥󠁮󠁧󠁿 **Premier League** (England)
    - 🇪🇸 **La Liga** (Spain)
    - 🇮🇹 **Serie A** (Italy)
    - 🇮🇹 **Serie B** (Italy)
    - 🇩🇪 **2.Bundesliga** (Germany)
    
    ### 📈 Data Coverage
    Seasons: 2023-24, 2024-25, 2025-26
    """)

else:
    df = st.session_state.df
    predictor = st.session_state.predictor
    
    # Get teams
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    
    st.header(f"🎯 Match Prediction - {selected_league}")
    
    # Team selection
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox("🏠 Home Team", teams, key='home')
    
    with col2:
        away_teams = [t for t in teams if t != home_team]
        away_team = st.selectbox("✈️ Away Team", away_teams, key='away')
    
    # Predict button
    if st.button("📊 Predict Match", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            predictions = predictor.predict_match(home_team, away_team, df, ml_weight)
            
            if predictions:
                st.markdown("---")
                
                # Match Result
                st.subheader("🏆 Match Result")
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.metric(
                        "Home Win",
                        f"{predictions['match_result']['home_win']*100:.1f}%"
                    )
                
                with c2:
                    st.metric(
                        "Draw",
                        f"{predictions['match_result']['draw']*100:.1f}%"
                    )
                
                with c3:
                    st.metric(
                        "Away Win",
                        f"{predictions['match_result']['away_win']*100:.1f}%"
                    )
                
                # Expected Goals
                st.subheader("⚽ Expected Goals")
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.metric(home_team, f"{predictions['expected_goals']['home']:.2f}")
                
                with c2:
                    st.metric("Total", f"{predictions['expected_goals']['total']:.2f}")
                
                with c3:
                    st.metric(away_team, f"{predictions['expected_goals']['away']:.2f}")
                
                # Betting Markets
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'over_2_5' in predictions:
                        st.subheader("📊 Over/Under 2.5 Goals")
                        over_prob = predictions['over_2_5']['probability']
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("Over 2.5", f"{over_prob*100:.1f}%")
                        with c2:
                            st.metric("Under 2.5", f"{(1-over_prob)*100:.1f}%")
                        
                        # Show ML vs MC breakdown
                        with st.expander("Model Breakdown"):
                            st.write(f"ML Model: {predictions['over_2_5']['ml_prob']*100:.1f}%")
                            st.write(f"Monte Carlo: {predictions['over_2_5']['mc_prob']*100:.1f}%")
                            st.write(f"Hybrid (weight={ml_weight:.0%}): {over_prob*100:.1f}%")
                
                with col2:
                    if 'btts' in predictions:
                        st.subheader("🎯 Both Teams to Score")
                        btts_prob = predictions['btts']['probability']
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("Yes", f"{btts_prob*100:.1f}%")
                        with c2:
                            st.metric("No", f"{(1-btts_prob)*100:.1f}%")
                        
                        # Show ML vs MC breakdown
                        with st.expander("Model Breakdown"):
                            st.write(f"ML Model: {predictions['btts']['ml_prob']*100:.1f}%")
                            st.write(f"Monte Carlo: {predictions['btts']['mc_prob']*100:.1f}%")
                            st.write(f"Hybrid (weight={ml_weight:.0%}): {btts_prob*100:.1f}%")
                
                # Asian Handicap
                st.subheader("📊 Asian Handicap")
                hcp_df = pd.DataFrame(predictions['asian_handicap'])
                hcp_df['Handicap'] = hcp_df['line'].apply(lambda x: f"{x:+.1f}" if x != 0 else "0.0")
                hcp_df = hcp_df[['Handicap', 'home', 'away']]
                hcp_df.columns = ['Handicap', f'{home_team} %', f'{away_team} %']
                
                st.dataframe(
                    hcp_df,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # Goal Lines
                st.subheader("🎯 Over/Under Goal Lines")
                gl_df = pd.DataFrame(predictions['goal_lines'])
                gl_df['Goals'] = gl_df['line'].apply(lambda x: f"{x:.1f}")
                gl_df = gl_df[['Goals', 'over', 'under']]
                gl_df.columns = ['Line', 'Over %', 'Under %']
                
                st.dataframe(
                    gl_df,
                    use_container_width=True,
                    hide_index=True,
                    height=350
                )
                
                # Value Betting Analysis
                st.subheader("💰 Value Betting")
                st.info("""
                **How to use these predictions:**
                - Compare probabilities with bookmaker odds
                - Look for bets where model probability > implied probability from odds
                - Example: If model says 65% but odds imply 55%, that's value
                - Always practice responsible bankroll management (1-2% per bet)
                """)
            
            else:
                st.error("Unable to make prediction")

# Footer
st.markdown("---")
st.markdown("""
**Data Source:** [football-data.co.uk](https://www.football-data.co.uk/)  
**Disclaimer:** For educational and entertainment purposes only. Gamble responsibly.
""")
