"""Data pipeline for real-time market data and sentiment analysis."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Market data snapshot."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    indicators: Dict[str, float]


@dataclass
class SentimentData:
    """Sentiment data from news and social media."""
    symbol: str
    timestamp: datetime
    sentiment_score: float  # -1 to 1
    positive_count: int
    negative_count: int
    neutral_count: int
    headline_count: int
    source: str  # "news", "reddit", "twitter", etc.


class DataSourceConnector:
    """Abstract data source connector."""
    
    async def fetch_market_data(self, symbol: str, timeframe: str = "1h") -> List[MarketData]:
        """Fetch market data."""
        raise NotImplementedError()
    
    async def fetch_sentiment(self, symbol: str) -> SentimentData:
        """Fetch sentiment data."""
        raise NotImplementedError()


class YFinanceConnector(DataSourceConnector):
    """Yahoo Finance data connector."""
    
    async def fetch_market_data(self, symbol: str, timeframe: str = "1h") -> List[MarketData]:
        """Fetch market data from Yahoo Finance."""
        try:
            import yfinance as yf
            
            # Map timeframe to yfinance interval
            interval_map = {
                "1m": "1m",
                "5m": "5m",
                "15m": "15m",
                "1h": "1h",
                "1d": "1d",
            }
            interval = interval_map.get(timeframe, "1h")
            
            ticker = yf.Ticker(symbol.replace("/", ""))
            hist = ticker.history(period="7d", interval=interval)
            
            market_data = []
            for idx, row in hist.iterrows():
                md = MarketData(
                    symbol=symbol,
                    timestamp=idx,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                    indicators={},
                )
                market_data.append(md)
            
            logger.info(f"Fetched {len(market_data)} candles for {symbol}")
            return market_data
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            return []
    
    async def fetch_sentiment(self, symbol: str) -> SentimentData:
        """Fetch sentiment from Yahoo Finance news."""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol.replace("/", ""))
            news = ticker.news
            
            # Simple sentiment analysis
            positive_words = ['surge', 'rally', 'bull', 'gain', 'jump', 'profit']
            negative_words = ['crash', 'plunge', 'bear', 'loss', 'fall', 'decline']
            
            positive_count = 0
            negative_count = 0
            
            for article in news[:10]:
                title = article.get('title', '').lower()
                for word in positive_words:
                    if word in title:
                        positive_count += 1
                for word in negative_words:
                    if word in title:
                        negative_count += 1
            
            sentiment_score = (positive_count - negative_count) / max(positive_count + negative_count, 1)
            
            return SentimentData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                sentiment_score=sentiment_score,
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=len(news) - positive_count - negative_count,
                headline_count=len(news),
                source="yahoo_finance",
            )
        except Exception as e:
            logger.error(f"Failed to fetch sentiment: {e}")
            return SentimentData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                sentiment_score=0.0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                headline_count=0,
                source="error",
            )


class AlpacaConnector(DataSourceConnector):
    """Alpaca data connector."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def fetch_market_data(self, symbol: str, timeframe: str = "1h") -> List[MarketData]:
        """Fetch market data from Alpaca."""
        try:
            import httpx
            
            # Map timeframe to Alpaca timeframe
            tf_map = {
                "1m": "1Min",
                "5m": "5Min",
                "15m": "15Min",
                "1h": "1Hour",
                "1d": "1Day",
            }
            tf = tf_map.get(timeframe, "1Hour")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://data.alpaca.markets/v1beta3/crypto/us/bars",
                    params={
                        "symbols": symbol,
                        "timeframe": tf,
                        "limit": 100,
                    },
                    headers={"APCA-API-KEY-ID": self.api_key},
                    timeout=30.0,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    market_data = []
                    
                    for bar in data.get("bars", {}).get(symbol, []):
                        md = MarketData(
                            symbol=symbol,
                            timestamp=datetime.fromisoformat(bar["t"]),
                            open=float(bar["o"]),
                            high=float(bar["h"]),
                            low=float(bar["l"]),
                            close=float(bar["c"]),
                            volume=float(bar["v"]),
                            indicators={},
                        )
                        market_data.append(md)
                    
                    logger.info(f"Fetched {len(market_data)} bars from Alpaca for {symbol}")
                    return market_data
                else:
                    logger.error(f"Alpaca API error: {response.status_code}")
                    return []
        except Exception as e:
            logger.error(f"Failed to fetch Alpaca data: {e}")
            return []
    
    async def fetch_sentiment(self, symbol: str) -> SentimentData:
        """Fetch sentiment from Alpaca news endpoint."""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://data.alpaca.markets/v1beta3/news",
                    params={"symbols": symbol, "limit": 20},
                    headers={"APCA-API-KEY-ID": self.api_key},
                    timeout=30.0,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get("news", [])
                    
                    positive_count = sum(1 for a in articles if a.get("sentiment") == "positive")
                    negative_count = sum(1 for a in articles if a.get("sentiment") == "negative")
                    neutral_count = len(articles) - positive_count - negative_count
                    
                    sentiment_score = (positive_count - negative_count) / max(len(articles), 1)
                    
                    return SentimentData(
                        symbol=symbol,
                        timestamp=datetime.utcnow(),
                        sentiment_score=sentiment_score,
                        positive_count=positive_count,
                        negative_count=negative_count,
                        neutral_count=neutral_count,
                        headline_count=len(articles),
                        source="alpaca_news",
                    )
                else:
                    logger.error(f"Alpaca news API error: {response.status_code}")
                    return SentimentData(
                        symbol=symbol,
                        timestamp=datetime.utcnow(),
                        sentiment_score=0.0,
                        positive_count=0,
                        negative_count=0,
                        neutral_count=0,
                        headline_count=0,
                        source="error",
                    )
        except Exception as e:
            logger.error(f"Failed to fetch Alpaca sentiment: {e}")
            return SentimentData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                sentiment_score=0.0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                headline_count=0,
                source="error",
            )


class DataPipeline:
    """Unified async data pipeline."""
    
    def __init__(self, primary_connector: DataSourceConnector, backup_connector: Optional[DataSourceConnector] = None):
        self.primary = primary_connector
        self.backup = backup_connector
        self.cache: Dict[str, MarketData] = {}
        self.sentiment_cache: Dict[str, SentimentData] = {}
    
    async def fetch_market_data(self, symbols: List[str], timeframe: str = "1h") -> Dict[str, List[MarketData]]:
        """Fetch market data for multiple symbols."""
        results = {}
        
        tasks = [self.primary.fetch_market_data(symbol, timeframe) for symbol in symbols]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, response in zip(symbols, responses):
            if isinstance(response, Exception):
                logger.error(f"Error fetching {symbol}: {response}")
                results[symbol] = []
            else:
                results[symbol] = response
                # Update cache
                if response:
                    self.cache[symbol] = response[-1]
        
        return results
    
    async def fetch_sentiment_data(self, symbols: List[str]) -> Dict[str, SentimentData]:
        """Fetch sentiment data for multiple symbols."""
        results = {}
        
        tasks = [self.primary.fetch_sentiment(symbol) for symbol in symbols]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, response in zip(symbols, responses):
            if isinstance(response, Exception):
                logger.error(f"Error fetching sentiment for {symbol}: {response}")
                results[symbol] = None
            else:
                results[symbol] = response
                self.sentiment_cache[symbol] = response
        
        return results
    
    async def start_streaming(self, symbols: List[str], update_interval: int = 300):
        """Start continuous data streaming."""
        logger.info(f"Starting data stream for {len(symbols)} symbols, update interval: {update_interval}s")
        
        while True:
            try:
                market_data = await self.fetch_market_data(symbols, "1h")
                sentiment_data = await self.fetch_sentiment_data(symbols)
                
                logger.info(f"Updated market and sentiment data for {len(symbols)} symbols")
                
                await asyncio.sleep(update_interval)
            except Exception as e:
                logger.error(f"Error in data streaming: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
