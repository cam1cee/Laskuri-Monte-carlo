# ⚽ Football Betting Predictor

**AI-powered betting predictions using Monte Carlo simulation**

## 🎯 Features

### 📊 Multi-Market Predictions

* **Goals:** Over/Under 2.5 & Both Teams To Score (BTTS)
* **Corners:** Over/Under 10.5 corners
* **Cards:** Over/Under 4.5 total cards
* **Asian Handicap:** Lines from -3.0 to +3.0
* **Goal Lines:** Over/Under from 0.5 to 6.0
* **Match Result:** Home/Draw/Away probabilities
* **Expected Goals:** Statistical predictions

### 💰 Value Betting

* **Expected Value (EV)** calculation for each market
* Automatically identifies value bets
* Customizable confidence and EV thresholds
* Save and track your bets

### 📈 Smart Analysis

* **Recent Form:** W/D/L visualization with color coding
* **League Context:** Uses proper league averages for accuracy
* **Team Statistics:** Home/away splits
* **Fatigue Analysis:** Days since last match

### 💾 Bet Tracking

* Save bets with odds and stake
* Track results (Won/Lost/Pending)
* View betting history
* Calculate win rate and ROI

## 🌍 Supported Leagues

* 🏴󠁧󠁢󠁥󠁮󠁧󠁿 **Premier League** (England)
* 🇪🇸 **La Liga** (Spain)
* 🇮🇹 **Serie A** (Italy)
* 🇮🇹 **Serie B** (Italy)
* 🇩🇪 **2.Bundesliga** (Germany)

## 📈 Data Coverage

* **Seasons:** 2023-24, 2024-25, 2025-26
* **Source:** [football-data.co.uk](https://www.football-data.co.uk/)
* **Features:** Goals, corners, cards, form

## 🚀 Quick Start

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with main file: `app.py`

## 🎓 How It Works

### 1. League Context

Uses proper league averages to calculate attack/defense strengths:
- Home team attack strength vs league average
- Away team attack strength vs league average
- Defense strengths calculated similarly

### 2. Monte Carlo Simulation

* 10,000 Poisson simulations per prediction
* Separate simulations for goals, corners, and cards
* Probability distributions for all markets

### 3. Expected Value

Compares model probability vs bookmaker odds:
- EV = (Probability × (Odds - 1)) - (1 - Probability)
- Positive EV indicates potential value
- Recommends bets with EV > 5% and confidence > 60%

## 📊 Betting Markets

### Over/Under 2.5 Goals
Probability the match has more/less than 2.5 total goals

### Both Teams to Score (BTTS)
Probability both teams score at least one goal

### Corners Over/Under 10.5
Based on team corner averages

### Cards Over/Under 4.5
Based on team yellow card averages

### Asian Handicap
Coverage probabilities for handicap lines:
- Negative: Home gives goals (favorite)
- Positive: Home receives goals (underdog)
- Zero: Draw no bet

### Goal Lines
Over/Under for various goal totals (0.5 to 6.0)

## 💡 Usage Tips

### Finding Value

1. Model probability should be higher than implied odds
2. Look for EV > 5% and confidence > 60%
3. Markets with less attention (corners, cards) often have more value

### Interpreting Form

* 🟢 = Win
* 🟡 = Draw
* 🔴 = Loss

Recent form shows last 5 games, most recent first.

### Bankroll Management

* Never bet more than 1-2% of bankroll per bet
* Track all bets to measure performance
* Focus on quality over quantity
* Use the bet tracker to stay disciplined

## 🔧 Settings

### Simulations
Choose between 5,000, 10,000, or 20,000 simulations:
- More simulations = more accurate but slower
- 10,000 is recommended default

### Value Betting Thresholds

**Min Confidence %:** Minimum probability to recommend (default 60%)
**Min EV %:** Minimum expected value to recommend (default 5%)

## 📂 Project Structure

```
├── app.py              # Complete Streamlit app
├── requirements.txt    # Dependencies
├── README.md          # This file
└── bets.json          # Saved bets (auto-created)
```

## ⚠️ Disclaimer

**For educational and entertainment purposes only.**

* This tool provides statistical predictions, not guarantees
* Gambling involves risk - never bet more than you can afford to lose
* Past performance does not indicate future results
* Always gamble responsibly
* Check your local laws regarding sports betting
* The authors are not responsible for any losses

## 🙏 Acknowledgments

* Data: [football-data.co.uk](https://www.football-data.co.uk/)
* Built with [Streamlit](https://streamlit.io)
* Inspired by [TommiKi/football_predict](https://github.com/TommiKi/football_predict)

---

**Made with ⚽ and 🤖**

*Bet smart, bet small, bet for fun.*
