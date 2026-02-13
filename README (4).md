# ⚽ Football Predictions - Monte Carlo Simulation

Advanced football match prediction app using Monte Carlo simulations.

## Features

- **Monte Carlo Simulations** - 5,000 to 50,000 simulations per match
- **Win Probabilities** - Home/Draw/Away percentages
- **Expected Goals** - Statistical goal predictions
- **Asian Handicap** - Lines from -3.0 to +3.0
- **Over/Under Goals** - Lines from 0.5 to 6.0
- **Both Teams to Score** - BTTS probabilities

## Leagues

- 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League
- 🇪🇸 La Liga
- 🇮🇹 Serie A
- 🇮🇹 Serie B
- 🇩🇪 2.Bundesliga

## Data Coverage

Seasons: 2023-24, 2024-25, 2025-26
Source: [football-data.co.uk](https://www.football-data.co.uk/)

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your repository
4. Set main file: `app.py`

## How It Works

### Monte Carlo Method

1. **Calculate Expected Goals**
   - Uses team attack/defense strengths vs league averages
   - Home advantage factored in

2. **Run Simulations**
   - Generates thousands of random match outcomes using Poisson distribution
   - Each simulation produces a scoreline

3. **Calculate Probabilities**
   - Win/Draw/Loss from simulation results
   - Asian Handicap coverage from goal differences
   - Over/Under from total goals

### Asian Handicap Logic

- **Negative handicap** (e.g., -1.0): Home team gives goals to away team
  - Home covers if they win by MORE than the handicap
- **Positive handicap** (e.g., +1.0): Home team receives goals
  - Home covers if they don't lose by MORE than the handicap
- **Push**: When result equals handicap, stake refunded (50% to each side)

### Example

Arsenal vs Brentford:
- Expected: Arsenal 2.44, Brentford 0.96
- Simulations show: Arsenal win 69.8%, Draw 17.1%, Brentford 13.0%
- Asian Handicap -1.0: Arsenal must win by 2+ goals = ~40-50%
- Asian Handicap +3.0 for Arsenal: Almost never lose by 4+ = ~99%

## Requirements

- Python 3.8+
- Internet connection (to fetch data)

## License

For educational purposes only.
