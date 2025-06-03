# 🧠 Multi-Agent Portfolio Recommender

A real-time investing simulator powered by multi-agent architecture. This application provides intelligent stock analysis, forecasts, and personalized investment recommendations using a combination of financial APIs, AI models, and interactive chat features.

## 🚀 Features

- 📈 **Real-Time Stock Analysis** – Analyze any stock symbol for up-to-date trends, price movement, and indicators  
- 🤖 **Multi-Agent Architecture** – Modular agents for forecasting, recommendation, sentiment analysis, and education  
- 🧮 **Forecasting Engine** – Uses LightGBM and Monte Carlo Simulation for probabilistic return modeling  
- 📊 **News Sentiment Agent** – Evaluates news headlines for sentiment and relevance to portfolio strategy  
- 💬 **Chat-Based Assistant** – Natural language interface to explain financial concepts and provide guidance  
- 📚 **Education Mode** – Learn about P/E ratios, diversification, risk profiles, and investment strategies  
- 📂 **Session History & Quick Access** – Keep track of investment questions and revisit key insights

## 🧰 Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python
- **APIs Used**:
  - Gemini
  - Finnhub
  - NewsAPI
  - GNews
  - GroqAPI

## 🧠 Agent System

| Agent             | Functionality                                           |
|------------------|---------------------------------------------------------|
| Forecast Agent    | Predicts short-term trends using time-series models     |
| Recommender Agent | Provides buy/sell/hold advice using model outputs       |
| Sentiment Agent   | Analyzes recent news sentiment around selected stocks   |
| Education Agent   | Answers user questions about investing concepts         |

## 🛠 Setup Instructions

1. **Clone this repo**
   ```bash
   git clone https://github.com/your-username/multi-agent-portfolio-recommender.git
   cd multi-agent-portfolio-recommender
