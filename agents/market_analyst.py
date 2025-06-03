from langchain_groq import ChatGroq
from utils.config import GROQ_API_KEY, NEWSAPI_KEY, FINNHUB_API_KEY, ALPHA_VANTAGE_API_KEY, FRED_API_KEY
from utils.logger import logger
import finnhub
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from cachetools import TTLCache
from typing import Dict, List, Optional
import json
import requests

class MarketAnalystAgent:
    def __init__(self):
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
        self.finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_KEY)
        self.cache = TTLCache(maxsize=100, ttl=3600)

    def analyze(self, user_message: str) -> str:
        """Analyze user message and provide market insights."""
        try:
            # Extract potential stock symbols from message
            words = user_message.upper().split()
            potential_symbols = [word for word in words if word.isalpha() and 2 <= len(word) <= 5]
            
            response_parts = []
            
            # If specific stocks are mentioned, analyze them in detail
            if potential_symbols:
                for symbol in potential_symbols[:3]:
                    try:
                        analysis = self.analyze_stock(symbol)
                        if analysis:
                            # Get company-specific news
                            market_news = self.fetch_market_news()
                            company_news = [news for news in market_news if symbol.lower() in news['title'].lower() or analysis['company'].lower() in news['title'].lower()]
                            
                            response_parts.append(f"\n**{analysis['company']} ({symbol}) Analysis:**")
                            response_parts.append(f"Current Price: ${analysis['price']:.2f}")
                            
                            if company_news:
                                response_parts.append("\nLatest Company News:")
                                for news in company_news[:3]:
                                    response_parts.append(f"- {news['title']} (Sentiment: {news['sentiment'].title()})")
                            
                            # Add financial metrics if available
                            if analysis['pe_ratio']:
                                response_parts.append(f"\nFinancial Metrics:")
                                response_parts.append(f"- P/E Ratio: {analysis['pe_ratio']}")
                            if analysis['debt_to_equity']:
                                response_parts.append(f"- Debt/Equity: {analysis['debt_to_equity']}")
                            
                            # Add market sentiment
                            response_parts.append(f"\nMarket Sentiment:")
                            response_parts.append(f"- News Sentiment: {analysis['news_sentiment']}")
                            
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {str(e)}")
            else:
                # If no specific stocks mentioned, provide general market overview
                prompt = f"""Based on the latest market news, provide a brief market overview focusing on:
                1. Key market movements
                2. Major sector trends
                3. Notable market events
                Keep it concise and focused on current market conditions."""
                
                response = self.llm.invoke(prompt)
                response_parts.append("\n**Market Overview:**")
                response_parts.append(response.content.strip())
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return "I apologize, but I encountered an error while analyzing the markets. Please try again or rephrase your question."

    def fetch_financials(self, cik: str) -> dict:
        cache_key = f"financials_{cik}"
        if cache_key in self.cache:
            logger.info(f"Returning cached financials for CIK {cik}")
            return self.cache[cache_key]

        try:
            logger.info(f"Fetching MySQL financials for CIK {cik}")
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)

            five_years_ago = datetime.now() - timedelta(days=5*365)

            cursor.execute("""
                SELECT revenue, net_income, fiscal_date_ending
                FROM income_statements
                WHERE cik = %s AND fiscal_date_ending >= %s
                ORDER BY fiscal_date_ending DESC
            """, (cik, five_years_ago))
            income = cursor.fetchall() or []
            logger.info(f"Income statements for CIK {cik}: {len(income)} records")

            cursor.execute("""
                SELECT total_assets, total_liabilities, total_equity
                FROM balance_sheets
                WHERE cik = %s AND fiscal_date_ending >= %s
                ORDER BY fiscal_date_ending DESC
            """, (cik, five_years_ago))
            balance = cursor.fetchall() or []
            logger.info(f"Balance sheets for CIK {cik}: {len(balance)} records")

            cursor.execute("""
                SELECT operating_cash_flow, capital_expenditure
                FROM cash_flows
                WHERE cik = %s AND fiscal_date_ending >= %s
                ORDER BY fiscal_date_ending DESC
            """, (cik, five_years_ago))
            cash_flow = cursor.fetchall() or []
            logger.info(f"Cash flows for CIK {cik}: {len(cash_flow)} records")

            cursor.close()
            conn.close()

            financials = {
                "income": income,
                "balance": balance,
                "cash_flow": cash_flow
            }
            self.cache[cache_key] = financials
            return financials

        except Exception as e:
            logger.error(f"Failed to fetch financials for CIK {cik}: {str(e)}")
            return {}

    def fetch_news_sentiment(self, symbols: List[str]) -> Dict[str, str]:
        sentiments = {}
        to_date = datetime.now()
        from_date = to_date - timedelta(days=7)

        for symbol in symbols:
            cache_key = f"news_{symbol}"
            if cache_key in self.cache:
                logger.info(f"Returning cached news sentiment for {symbol}")
                sentiments[symbol] = self.cache[cache_key]
                continue

            try:
                logger.info(f"Fetching news for {symbol}")
                response = self.newsapi_client.get_everything(
                    q=symbol,
                    from_param=from_date.strftime('%Y-%m-%d'),
                    to=to_date.strftime('%Y-%m-%d'),
                    language='en',
                    sort_by='relevancy'
                )
                articles = response.get('articles', [])
                if not articles:
                    logger.info(f"No news articles found for {symbol}")
                    sentiments[symbol] = "Neutral"
                    self.cache[cache_key] = "Neutral"
                    continue

                headlines = [article['title'] for article in articles[:5]]
                prompt = f"""
Analyze the sentiment of these news headlines for {symbol}:
{headlines}
Score sentiment from -1 (negative) to 1 (positive). Return a JSON object:
```json
{{
    "sentiment": "Positive",
    "score": 0.8
}}
Where sentiment is 'Positive' (>=0.3), 'Negative' (<= -0.3), or 'Neutral' (else).
"""
                response = self.llm.invoke(prompt)
                result = json.loads(response.content.strip())
                sentiment = result.get("sentiment", "Neutral")

                if sentiment not in ["Positive", "Negative", "Neutral"]:
                    logger.warning(f"Invalid sentiment for {symbol}: {sentiment}")
                    sentiment = "Neutral"

                logger.info(f"News sentiment for {symbol}: {sentiment}")
                sentiments[symbol] = sentiment
                self.cache[cache_key] = sentiment

            except Exception as e:
                logger.error(f"Failed to fetch news for {symbol}: {str(e)}")
                if "429" in str(e):
                    time.sleep(10)
                sentiments[symbol] = "Neutral"
                self.cache[cache_key] = "Neutral"

        return sentiments

    def calculate_ratios(self, financials: dict, current_price: float, shares_outstanding: float) -> dict:
        try:
            latest_income = financials.get("income", [{}])[0]
            latest_balance = financials.get("balance", [{}])[0]

            eps = latest_income.get("net_income", 0) / shares_outstanding if shares_outstanding else 0
            pe_ratio = current_price / eps if eps != 0 else None
            debt_to_equity = (
                latest_balance.get("total_liabilities", 0) /
                latest_balance.get("total_equity", 1)
                if latest_balance.get("total_equity", 0) != 0 else None
            )

            return {
                "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
                "debt_to_equity": round(debt_to_equity, 2) if debt_to_equity else None
            }
        except Exception as e:
            logger.error(f"Failed to calculate ratios: {str(e)}")
            return {"pe_ratio": None, "debt_to_equity": None}

    def _generate_market_analysis(self, data: dict, prompt: str) -> str:
        """Generate market analysis using ChatGPT."""
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating market analysis: {str(e)}")
            return "Unable to generate analysis at this time."

    def analyze_stock(self, symbol: str) -> dict:
        """Analyze stock with enhanced GPT analysis."""
        try:
            # Get basic stock data
            quote = self.finnhub_client.quote(symbol)
            company = self.finnhub_client.company_profile2(symbol=symbol)
            
            # Get news sentiment
            news = self.fetch_market_news()
            relevant_news = [n for n in news if symbol.lower() in n['title'].lower()]
            
            # Prepare data for analysis
            stock_data = {
                "symbol": symbol,
                "company": company.get("name", symbol),
                "current_price": quote.get("c", 0.0),
                "daily_change": quote.get("d", 0.0),
                "daily_change_percent": quote.get("dp", 0.0),
                "high": quote.get("h", 0.0),
                "low": quote.get("l", 0.0),
                "news": relevant_news[:2]
            }
            
            # Generate concise analysis
            analysis_prompt = "Provide a brief, factual summary of the stock's current state."
            analysis = self._generate_market_analysis(stock_data, analysis_prompt)
            
            stock_data["analysis"] = analysis
            return stock_data
            
        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "error": f"Unable to analyze {symbol} at this time."
            }

    def fetch_market_news(self) -> List[Dict]:
        """Fetch real-time market news from Alpha Vantage."""
        cache_key = "market_news"
        if cache_key in self.cache:
            logger.info("Returning cached market news")
            return self.cache[cache_key]

        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={ALPHA_VANTAGE_API_KEY}&topics=financial_markets,technology,economy_fiscal,economy_monetary"
            response = requests.get(url)
            data = response.json()
            
            if "feed" not in data:
                logger.error("No news feed in Alpha Vantage response")
                return []
                
            news_items = []
            for item in data["feed"][:10]:  # Get latest 10 news items
                news_items.append({
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "sentiment": item.get("overall_sentiment_label", "neutral"),
                    "time_published": item.get("time_published", "")
                })
            
            logger.info(f"Fetched {len(news_items)} news items from Alpha Vantage")
            self.cache[cache_key] = news_items
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching market news: {str(e)}")
            return []

    def fetch_fred_data(self, series_id: str, start_date: Optional[str] = None) -> Dict:
        """Fetch economic data from FRED."""
        cache_key = f"fred_{series_id}"
        if cache_key in self.cache:
            logger.info(f"Returning cached FRED data for {series_id}")
            return self.cache[cache_key]

        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "observation_start": start_date,
                "sort_order": "desc"
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if "observations" not in data:
                logger.error(f"No observations found for FRED series {series_id}")
                return {}
                
            result = {
                "values": [float(obs["value"]) for obs in data["observations"] if obs["value"] != "."],
                "dates": [obs["date"] for obs in data["observations"] if obs["value"] != "."]
            }
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error fetching FRED data for {series_id}: {str(e)}")
            return {}

    def get_economic_indicators(self) -> Dict:
        """Get key economic indicators from FRED."""
        indicators = {
            "GDP": "GDP",              # Real GDP
            "UNRATE": "UNRATE",        # Unemployment Rate
            "CPIAUCSL": "CPIAUCSL",    # Consumer Price Index
            "DFF": "DFF",              # Federal Funds Rate
            "T10Y2Y": "T10Y2Y",        # 10-Year Treasury Constant Maturity Minus 2-Year
            "MORTGAGE30US": "MORTGAGE30US"  # 30-Year Fixed Rate Mortgage Average
        }
        
        economic_data = {}
        for indicator_name, series_id in indicators.items():
            data = self.fetch_fred_data(series_id)
            if data and data.get("values"):
                economic_data[indicator_name] = {
                    "current": data["values"][0],
                    "previous": data["values"][1] if len(data["values"]) > 1 else None,
                    "trend": "up" if len(data["values"]) > 1 and data["values"][0] > data["values"][1] else "down"
                }
        
        return economic_data

    def _format_stock_analysis(self, stock_data: dict, timeframe: str) -> list:
        """Format detailed stock analysis."""
        parts = []
        parts.append(f"\n**{stock_data['company']} ({stock_data['symbol']}) Analysis:**")
        parts.append(f"Current Price: ${stock_data['current_price']:.2f}")
        
        if stock_data.get('pe_ratio'):
            parts.append(f"P/E Ratio: {stock_data['pe_ratio']:.2f}")
        if stock_data.get('debt_to_equity'):
            parts.append(f"Debt/Equity: {stock_data['debt_to_equity']:.2f}")
            
        # Add recent performance
        parts.append(f"\nPerformance ({timeframe}):")
        # Add performance metrics based on timeframe
        
        # Add sentiment analysis
        parts.append(f"\nMarket Sentiment: {stock_data['news_sentiment']}")
        
        return parts

    def _format_stock_comparison(self, stocks_data: list) -> list:
        """Format comparison between multiple stocks."""
        parts = []
        parts.append("\n**Stock Comparison:**")
        
        # Create comparison table
        comparison_points = ["Current Price", "P/E Ratio", "Debt/Equity", "News Sentiment"]
        for point in comparison_points:
            parts.append(f"\n{point}:")
            for stock in stocks_data:
                if point == "Current Price":
                    parts.append(f"- {stock['symbol']}: ${stock['current_price']:.2f}")
                elif point == "P/E Ratio" and stock.get('pe_ratio'):
                    parts.append(f"- {stock['symbol']}: {stock['pe_ratio']:.2f}")
                elif point == "Debt/Equity" and stock.get('debt_to_equity'):
                    parts.append(f"- {stock['symbol']}: {stock['debt_to_equity']:.2f}")
                elif point == "News Sentiment":
                    parts.append(f"- {stock['symbol']}: {stock['news_sentiment']}")
        
        return parts

    def _format_technical_analysis(self, stock_data: dict, metrics: list) -> list:
        """Format technical analysis metrics."""
        parts = []
        parts.append(f"\n**Technical Analysis for {stock_data['symbol']}:**")
        
        # Add requested technical indicators
        for metric in metrics:
            if metric.lower() == "rsi":
                parts.append("RSI (14-day): [Calculate RSI]")
            elif metric.lower() == "moving_averages":
                parts.append("Moving Averages:")
                parts.append("- 50-day MA: [Calculate 50-day MA]")
                parts.append("- 200-day MA: [Calculate 200-day MA]")
        
        return parts

    def _format_news_analysis(self, symbol: str, news: list) -> list:
        """Format news analysis."""
        parts = []
        parts.append(f"\n**Recent News Impact on {symbol}:**")
        
        if news:
            for item in news[:3]:
                parts.append(f"- {item['title']}")
                parts.append(f"  Sentiment: {item['sentiment'].title()}")
        else:
            parts.append("No significant news found recently.")
        
        return parts

    def _analyze_sector_performance(self) -> list:
        """Analyze sector performance with economic context."""
        parts = []
        parts.append("\n**Sector Performance and Economic Context:**")
        
        # Get economic indicators
        economic_data = self.get_economic_indicators()
        
        if economic_data:
            parts.append("\nKey Economic Indicators:")
            if "UNRATE" in economic_data:
                parts.append(f"- Unemployment Rate: {economic_data['UNRATE']['current']}% ({economic_data['UNRATE']['trend']})")
            if "CPIAUCSL" in economic_data:
                parts.append(f"- Consumer Price Index: {economic_data['CPIAUCSL']['current']} ({economic_data['CPIAUCSL']['trend']})")
            if "DFF" in economic_data:
                parts.append(f"- Federal Funds Rate: {economic_data['DFF']['current']}% ({economic_data['DFF']['trend']})")
            if "MORTGAGE30US" in economic_data:
                parts.append(f"- 30-Year Mortgage Rate: {economic_data['MORTGAGE30US']['current']}% ({economic_data['MORTGAGE30US']['trend']})")
        
        # Generate sector analysis considering economic data
        analysis_prompt = f"""Based on these economic indicators:
        {economic_data}
        
        Provide a brief analysis of how these conditions might affect different market sectors.
        Focus on:
        1. Which sectors might benefit
        2. Which sectors might face challenges
        3. Key trends to watch
        Keep it concise and actionable."""
        
        response = self._generate_market_analysis(economic_data, analysis_prompt)
        parts.append("\nSector Analysis:")
        parts.append(response)
        
        return parts

    def _generate_market_overview(self) -> list:
        """Generate market overview with economic context."""
        parts = []
        parts.append("\n**Market Overview:**")
        
        # Get economic indicators
        economic_data = self.get_economic_indicators()
        
        # Get market news
        market_news = self.fetch_market_news()
        
        # Generate comprehensive market analysis
        analysis_prompt = f"""Based on:
        1. Economic Indicators: {economic_data}
        2. Recent Market News: {[news['title'] for news in market_news[:5]]}
        
        Provide a comprehensive market overview focusing on:
        1. Current market conditions
        2. Economic factors affecting markets
        3. Key risks and opportunities
        4. Short-term outlook
        Keep it concise and data-driven."""
        
        response = self._generate_market_analysis(
            {"economic_data": economic_data, "market_news": market_news},
            analysis_prompt
        )
        parts.append(response)
        
        return parts
