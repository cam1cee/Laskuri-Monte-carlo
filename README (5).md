# ⚽ Football Betting Predictor

**AI-powered betting predictions using Machine Learning + Monte Carlo simulation**

## 🎯 Features

### 📊 Multi-Market Predictions

* **Match Result:** Home/Draw/Away probabilities
* **Goals:** Over/Under 2.5 & Both Teams To Score (BTTS)
* **Asian Handicap:** Lines from -3.0 to +3.0
* **Goal Lines:** Over/Under from 0.5 to 6.0 goals
* **Expected Goals:** Statistical goal predictions

### 🌍 Supported Leagues

* 🏴󠁧󠁢󠁥󠁮󠁧󠁿 **Premier League** (England)
* 🇪🇸 **La Liga** (Spain)
* 🇮🇹 **Serie A** (Italy)
* 🇮🇹 **Serie B** (Italy)
* 🇩🇪 **2.Bundesliga** (Germany)

### 🤖 Dual Prediction System

* **Machine Learning:** Gradient Boosting + Random Forest models
* **Monte Carlo:** 10,000 Poisson simulations per prediction
* **Hybrid:** Weighted combination for optimal accuracy (default 60% ML / 40% MC)

### 📈 Data Coverage

* **Seasons:** 2023-24, 2024-25, 2025-26
* **Source:** [football-data.co.uk](https://www.football-data.co.uk/)
* **Features:** Goals, form, corners, cards (when available)

## 🚀 Quick Start

### Run Web App (Streamlit)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run ui.py
```

The app will open in your browser at `http://localhost:8501`

### Run Command Line

```bash
python predictor.py
```

Interactive mode - follow the prompts!

## 📱 Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your repository
4. Set main file: `ui.py`

## 🎓 How It Works

### 1. Data Collection

* Downloads historical match data from football-data.co.uk
* Covers last 3 seasons (2023-24, 2024-25, 2025-26)
* Automatically loads when you select a league

### 2. Feature Engineering

* Goals scored/conceded averages (home/away split)
* Recent form (win rate in last 5 games)
* Corners and cards averages (when available)
* Team volatility and consistency metrics

### 3. Machine Learning Models

* **Over/Under 2.5:** Gradient Boosting Classifier
* **BTTS:** Random Forest Classifier
* Trained on hundreds of matches per league
* Cross-validated for accuracy

### 4. Monte Carlo Simulation

* Poisson distribution for goals
* 10,000 iterations per prediction
* Probability distribution analysis
* Asian Handicap and Goal Lines calculation

### 5. Hybrid Prediction

* Combines ML model output with Monte Carlo simulation
* Default weight: 60% ML, 40% MC (adjustable in sidebar)
* Provides balanced, robust predictions

## 📊 Prediction Outputs

### Match Result
Win probabilities for Home, Draw, and Away

### Expected Goals
Statistical prediction of goals for each team

### Over/Under 2.5 Goals
Probability the match has more or less than 2.5 total goals

### Both Teams to Score (BTTS)
Probability both teams score at least one goal

### Asian Handicap
Coverage probabilities for handicap lines from -3.0 to +3.0
- Negative handicap: Home team gives goals (favorite)
- Positive handicap: Home team receives goals (underdog)

### Goal Lines
Over/Under probabilities for lines from 0.5 to 6.0 goals

## 💡 Usage Tips

### Finding Value

* Look for differences between model probability and bookmaker odds
* Example: Model says 65% Over 2.5, but odds imply only 55% → potential value
* Higher probability = higher confidence

### Understanding Asian Handicap

* **-1.0:** Home team must win by 2+ goals to cover
* **0.0:** Draw no bet (refund if draw)
* **+1.0:** Home team can lose by 1 and you still get refund (push)

### Bankroll Management

* Never bet more than 1-2% of bankroll per bet
* Focus on quality over quantity
* Track your bets to measure performance

## 🔧 Advanced Settings

### ML/MC Weight Slider

Adjust the balance between Machine Learning and Monte Carlo:
* **Higher ML weight (60-80%):** Trust statistical models more
* **Higher MC weight (40-60%):** Trust simulation more
* **Balanced (50/50):** Equal weighting

### Model Breakdown

Click "Model Breakdown" to see:
* ML Model probability
* Monte Carlo probability
* Final hybrid prediction

## 📂 Project Structure

```
├── ui.py                     # Streamlit web interface
├── predictor.py              # Core prediction engine
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🛠️ Technical Details

### Machine Learning

* **Scikit-learn** models
* Feature engineering with rolling averages
* Form calculation with time decay
* Train/test split validation

### Monte Carlo

* NumPy Poisson distribution
* 10,000 simulations per prediction
* Statistical probability calculation
* Asian Handicap logic implementation

### Data Processing

* Pandas for data manipulation
* Automatic caching of loaded data
* Error handling for missing data
* Support for multiple seasons

## ⚠️ Disclaimer

**For educational and entertainment purposes only.**

* This tool provides statistical predictions, not guarantees
* Gambling involves risk - never bet more than you can afford to lose
* Past performance does not indicate future results
* Always gamble responsibly
* Check your local laws regarding sports betting
* The authors are not responsible for any losses

## 🤝 Contributing

Contributions welcome! Feel free to:

* Report bugs
* Suggest features
* Submit pull requests
* Share your results

## 📄 License

MIT License - free to use and modify for personal use.

## 🙏 Acknowledgments

* Data source: [football-data.co.uk](https://www.football-data.co.uk/)
* Built with [Streamlit](https://streamlit.io)
* ML powered by [scikit-learn](https://scikit-learn.org)
* Inspired by [TommiKi/football_predict](https://github.com/TommiKi/football_predict)

---

**Made with ⚽ and 🤖**

*Remember: Bet smart, bet small, bet for fun.*
