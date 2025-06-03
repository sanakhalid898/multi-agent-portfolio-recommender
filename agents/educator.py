import requests
from utils.logger import logger
import json
import traceback
import time
import re
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.config import FINNHUB_API_KEY

class EducatorAgent:
    def __init__(self):
        self.api_available = False
        self.base_url = "http://localhost:11434/api"
        self.model = "gemma:2b"
        self.finnhub_url = "https://finnhub.io/api/v1"
        
        try:
            # Test Ollama connection
            logger.info(f"Testing Ollama connection with {self.model}...")
            response = self._generate_with_retry("Respond with 'OK' if you can read this message.")
            
            if response:
                self.api_available = True
                logger.info("Successfully connected to Ollama")
            else:
                raise Exception("No valid response from test call")
                
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.api_available = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _generate_with_retry(self, prompt):
        """Generate content with retry logic."""
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.warning(f"Error generating response: {str(e)}")
            raise

    def _get_company_info(self, company_name):
        """Get company information using Finnhub API."""
        try:
            # Basic mapping of common companies
            company_map = {
                'tesla': 'TSLA',
                'apple': 'AAPL',
                'microsoft': 'MSFT',
                'amazon': 'AMZN',
                'google': 'GOOGL',
                'meta': 'META',
                'netflix': 'NFLX',
                'nvidia': 'NVDA',
                'jp morgan': 'JPM',
                'jpmorgan': 'JPM'
            }
            
            # Get ticker symbol
            ticker_symbol = company_map.get(company_name.lower(), company_name.upper())
            
            # Fetch company profile
            profile_url = f"{self.finnhub_url}/stock/profile2"
            profile_params = {
                'symbol': ticker_symbol,
                'token': FINNHUB_API_KEY
            }
            profile_response = requests.get(profile_url, params=profile_params)
            profile_response.raise_for_status()
            profile_data = profile_response.json()
            
            # Fetch quote data
            quote_url = f"{self.finnhub_url}/quote"
            quote_params = {
                'symbol': ticker_symbol,
                'token': FINNHUB_API_KEY
            }
            quote_response = requests.get(quote_url, params=quote_params)
            quote_response.raise_for_status()
            quote_data = quote_response.json()
            
            # Calculate price change percentage
            price_change_percent = ((quote_data.get('c', 0) - quote_data.get('pc', 0)) / quote_data.get('pc', 1)) * 100 if quote_data.get('pc', 0) != 0 else 0
            
            return {
                'name': profile_data.get('name', ticker_symbol),
                'symbol': ticker_symbol,
                'sector': profile_data.get('finnhubIndustry', 'N/A'),
                'industry': profile_data.get('finnhubIndustry', 'N/A'),
                'current_price': quote_data.get('c', 'N/A'),
                'market_cap': profile_data.get('marketCapitalization', 'N/A'),
                'description': profile_data.get('description', 'N/A'),
                'recent_change': f"{price_change_percent:.2f}%",
                'volume': quote_data.get('v', 'N/A'),
                'high_today': quote_data.get('h', 'N/A'),
                'low_today': quote_data.get('l', 'N/A'),
                'open_price': quote_data.get('o', 'N/A'),
                'prev_close': quote_data.get('pc', 'N/A')
            }
        except Exception as e:
            logger.error(f"Error fetching company info for {company_name}: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def _extract_companies(self, text):
        """Extract company names or ticker symbols from text."""
        # Common company names and their variations
        company_patterns = {
            r'\b(tesla|tsla)\b': 'tesla',
            r'\b(apple|aapl)\b': 'apple',
            r'\b(microsoft|msft)\b': 'microsoft',
            r'\b(amazon|amzn)\b': 'amazon',
            r'\b(google|googl|goog)\b': 'google',
            r'\b(meta|facebook|fb)\b': 'meta',
            r'\b(netflix|nflx)\b': 'netflix',
            r'\b(nvidia|nvda)\b': 'nvidia',
            r'\b(jp morgan|jpmorgan|jpm)\b': 'jp morgan'
        }
        
        companies = []
        for pattern, company in company_patterns.items():
            if re.search(pattern, text.lower()):
                companies.append(company)
        
        # Also look for potential ticker symbols (4-5 uppercase letters)
        tickers = re.findall(r'\b[A-Z]{4,5}\b', text)
        companies.extend(tickers)
        
        return list(set(companies))  # Remove duplicates

    def _get_fallback_response(self, topic: str) -> str:
        """Provide basic responses when API is unavailable."""
        # Extract any company names
        companies = self._extract_companies(topic)
        
        # If we have company names, try to get their info even in fallback mode
        if companies:
            response = "While I'm having some trouble with my main knowledge base, I can tell you about the companies you mentioned:\n\n"
            for company in companies:
                info = self._get_company_info(company)
                if info:
                    response += f"📈 {info['name']} ({info['symbol']})\n"
                    response += f"Current Price: ${info['current_price']}\n"
                    response += f"Recent Performance: {info['recent_change']}\n"
                    response += f"Industry: {info['industry']}\n\n"
            return response

        # Otherwise, use the topic-based responses
        basic_responses = {
            "market": """Let me tell you about the stock market in simple terms:
            
            🏢 Think of it like a marketplace where people buy and sell pieces of companies (stocks).
            
            Key things to understand:
            - Stock prices go up and down based on supply and demand
            - Company performance, economic news, and global events all affect prices
            - Major indices like S&P 500 and NASDAQ help us track how the overall market is doing
            
            Would you like to know more about any specific aspect of the market?""",
            
            "investing": """Here's what you need to know about smart investing:
            
            🎯 The key is to have a clear strategy and stick to it:
            
            1. Diversification: Spread your investments across different types of assets
            2. Long-term thinking: Focus on growth over time rather than quick gains
            3. Research: Always understand what you're investing in
            4. Risk management: Never invest more than you can afford to lose
            
            What aspect of investing would you like to explore further?""",
            
            "trading": """Let me explain trading in a way that's easy to understand:
            
            💹 Trading is about buying and selling investments, usually over shorter time periods:
            
            1. Types of Orders:
               - Market orders: Buy/sell right away at current price
               - Limit orders: Set your own price targets
            
            2. Important Concepts:
               - Volume: How many shares are being traded
               - Price movements: How values change during the day
            
            Would you like me to explain any of these concepts in more detail?""",
            
            "strategy": """Let's talk about investment strategies that work:
            
            📊 Here are some popular approaches:
            
            1. Buy and Hold: Invest in good companies and keep them for the long term
            2. Dollar-Cost Averaging: Invest fixed amounts regularly
            3. Index Investing: Follow the market with index funds
            4. Dividend Investing: Focus on income-generating stocks
            
            Which strategy would you like to learn more about?""",
            
            "risk": """Let's discuss investment risks and how to handle them:
            
            ⚠️ Understanding risks is crucial for successful investing:
            
            1. Market Risk: Overall market movements
            2. Company Risk: Individual company performance
            3. Economic Risk: How the economy affects investments
            4. Liquidity Risk: How easily you can buy/sell
            
            Would you like to explore any of these risks in more detail?"""
        }
        
        # Try to match the topic with basic responses
        for key, response in basic_responses.items():
            if key.lower() in topic.lower():
                return response
        
        return """I'm currently operating in fallback mode, but I can still help! I can provide information about:

        📚 Investment Topics:
        - Market basics and how they work
        - Investment principles and strategies
        - Trading concepts and techniques
        - Risk management and analysis
        
        🏢 Company Analysis:
        - Just mention a company name or ticker symbol
        - I can provide current market data and basic analysis
        
        What would you like to learn about?"""

    def provide_education(self, user_message: str) -> str:
        """Main method to handle user queries and provide responses."""
        try:
            if not self.api_available:
                return self._get_fallback_response(user_message)

            # Extract any company names for additional context
            companies = self._extract_companies(user_message)
            company_info = ""
            
            if companies:
                for company in companies:
                    info = self._get_company_info(company)
                    if info:
                        company_info += f"\nCompany Information for {info['name']} ({info['symbol']}):"
                        company_info += f"\n- Current Price: ${info['current_price']}"
                        company_info += f"\n- Recent Performance: {info['recent_change']}"
                        company_info += f"\n- Industry: {info['industry']}\n"

            # Construct the prompt with company information if available
            system_prompt = """You are a knowledgeable financial advisor and educator. 
            Provide clear, accurate, and helpful information about investing and the stock market. 
            Focus on educational value and practical advice. 
            If discussing specific companies, use the provided company information."""

            prompt = f"{system_prompt}\n\nUser Question: {user_message}"
            if company_info:
                prompt += f"\n\nRelevant Company Information:{company_info}"

            response = self._generate_with_retry(prompt)
            return response if response else self._get_fallback_response(user_message)

        except Exception as e:
            logger.error(f"Error in provide_education: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._get_fallback_response(user_message)

    def _provide_definition(self, query_info: dict) -> list:
        """Provide definition and explanation of investment terms."""
        parts = []
        
        definition_prompt = f"""Provide a clear explanation of {query_info['topic']} for a {query_info['user_context']} investor.
        Include:
        1. Simple definition
        2. Real-world example
        3. Why it's important
        4. Common misconceptions
        Keep it concise and appropriate for their level."""
        
        try:
            response = self.model.generate_content(definition_prompt)
            
            parts.append(f"\n**Understanding {query_info['topic'].title()}:**")
            parts.append(response.text.strip())
            return parts
        except Exception as e:
            logger.error(f"Error in definition generation: {str(e)}")
            return [f"\n**Understanding {query_info['topic'].title()}:**", "I apologize, but I'm having trouble generating the definition right now. Please try again in a moment."]

    def _provide_how_to_guide(self, query_info: dict) -> list:
        """Provide step-by-step guidance."""
        parts = []
        
        guide_prompt = f"""Create a step-by-step guide for {query_info['topic']} appropriate for a {query_info['user_context']} investor.
        Include:
        1. Prerequisites
        2. Step-by-step instructions
        3. Common pitfalls to avoid
        4. Tools or resources needed"""
        
        response = self.model.generate_content(guide_prompt)
        
        parts.append(f"\n**How to {query_info['topic'].title()}:**")
        parts.append(response.text.strip())
        return parts

    def _provide_beginner_guide(self, query_info: dict) -> list:
        """Provide beginner-friendly guidance."""
        parts = []
        
        guide_prompt = f"""Create a beginner-friendly guide about {query_info['topic']}.
        Include:
        1. Basic concepts
        2. First steps to take
        3. Common terms they'll encounter
        4. Resources for learning more
        Keep it simple and encouraging."""
        
        response = self.model.generate_content(guide_prompt)
        
        parts.append("\n**Getting Started:**")
        parts.append(response.text.strip())
        return parts

    def _provide_strategy_advice(self, query_info: dict) -> list:
        """Provide investment strategy advice."""
        parts = []
        
        strategy_prompt = f"""Provide investment strategy advice for {query_info['topic']} considering:
        - Risk profile: {query_info['risk_profile']}
        - Time horizon: {query_info['time_horizon']}
        - Investment goals: {query_info['investment_goals']}
        Include:
        1. Strategy overview
        2. Key considerations
        3. Implementation steps
        4. Risk factors to consider"""
        
        response = self.model.generate_content(strategy_prompt)
        
        parts.append("\n**Investment Strategy:**")
        parts.append(response.text.strip())
        return parts

    def _provide_risk_guidance(self, query_info: dict) -> list:
        """Provide risk management guidance."""
        parts = []
        
        risk_prompt = f"""Provide risk management guidance for {query_info['topic']} considering:
        - Risk profile: {query_info['risk_profile']}
        - Investment goals: {query_info['investment_goals']}
        Include:
        1. Risk factors
        2. Mitigation strategies
        3. Warning signs to watch for
        4. Best practices"""
        
        response = self.model.generate_content(risk_prompt)
        
        parts.append("\n**Risk Management:**")
        parts.append(response.text.strip())
        return parts

    def _provide_portfolio_guidance(self, query_info: dict) -> list:
        """Provide portfolio building guidance."""
        parts = []
        
        portfolio_prompt = f"""Provide portfolio building guidance considering:
        - Risk profile: {query_info['risk_profile']}
        - Time horizon: {query_info['time_horizon']}
        - Investment goals: {query_info['investment_goals']}
        Include:
        1. Asset allocation strategy
        2. Diversification approach
        3. Rebalancing guidelines
        4. Monitoring tips"""
        
        response = self.model.generate_content(portfolio_prompt)
        
        parts.append("\n**Portfolio Building:**")
        parts.append(response.text.strip())
        return parts