"""
Market analyzer agent for oil and gas commodities.
This module implements an agentic AI for market analysis.
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.rag.agents.insight_qa_agent import InsightQAAgent, create_qa_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/market_analyzer_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarketAnalyzerAgent:
    """
    Agentic AI for market analysis of oil and gas commodities.
    """
    
    def __init__(
        self, 
        data_dir: str = 'data/processed',
        insights_dir: str = 'data/insights',
        qa_agent: Optional[InsightQAAgent] = None
    ):
        """
        Initialize the market analyzer agent.
        
        Parameters
        ----------
        data_dir : str, optional
            Directory containing processed data, by default 'data/processed'
        insights_dir : str, optional
            Directory for storing insights, by default 'data/insights'
        qa_agent : InsightQAAgent, optional
            QA agent for retrieving insights, by default None (will create a new one)
        """
        self.data_dir = data_dir
        self.insights_dir = insights_dir
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(insights_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Initialize QA agent
        self.qa_agent = qa_agent or create_qa_agent()
        
        # Load commodity data
        self.commodity_data = self._load_commodity_data()
    
    def _load_commodity_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load processed commodity data.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of commodity DataFrames
        """
        commodity_data = {}
        
        # Get list of parquet files
        parquet_files = [f for f in os.listdir(self.data_dir) if f.endswith('.parquet')]
        
        for file in parquet_files:
            try:
                # Extract commodity name
                commodity = file.split('.')[0]
                
                # Load data
                file_path = os.path.join(self.data_dir, file)
                df = pd.read_parquet(file_path)
                
                # Add to dictionary
                commodity_data[commodity] = df
                
                logger.info(f"Loaded {len(df)} rows for {commodity}")
                
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        return commodity_data
    
    def generate_market_summary(
        self, 
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate a market summary for all commodities.
        
        Parameters
        ----------
        lookback_days : int, optional
            Number of days to look back, by default 30
        
        Returns
        -------
        Dict[str, Any]
            Market summary
        """
        if not self.commodity_data:
            logger.warning("No commodity data available")
            return {}
        
        # Get current date
        current_date = datetime.now()
        
        # Calculate start date
        start_date = current_date - timedelta(days=lookback_days)
        
        # Initialize summary
        summary = {
            'date': current_date.strftime('%Y-%m-%d'),
            'lookback_days': lookback_days,
            'commodities': {}
        }
        
        # Generate summary for each commodity
        for commodity, df in self.commodity_data.items():
            try:
                # Filter data for lookback period
                if isinstance(df.index, pd.DatetimeIndex):
                    recent_data = df[df.index >= start_date]
                else:
                    logger.warning(f"Index for {commodity} is not DatetimeIndex")
                    recent_data = df.iloc[-lookback_days:]
                
                if recent_data.empty:
                    logger.warning(f"No recent data for {commodity}")
                    continue
                
                # Get price column (assume first column is price)
                price_col = df.columns[0]
                
                # Calculate statistics
                current_price = recent_data[price_col].iloc[-1]
                previous_price = recent_data[price_col].iloc[0]
                price_change = current_price - previous_price
                price_change_pct = price_change / previous_price
                
                # Calculate volatility
                volatility = recent_data[price_col].pct_change().std() * np.sqrt(252)  # Annualized
                
                # Calculate trend
                trend = 'up' if price_change > 0 else 'down' if price_change < 0 else 'flat'
                
                # Calculate momentum
                momentum = recent_data[price_col].diff(5).iloc[-1]  # 5-day momentum
                
                # Add to summary
                summary['commodities'][commodity] = {
                    'current_price': current_price,
                    'previous_price': previous_price,
                    'price_change': price_change,
                    'price_change_pct': price_change_pct,
                    'volatility': volatility,
                    'trend': trend,
                    'momentum': momentum
                }
                
            except Exception as e:
                logger.error(f"Error generating summary for {commodity}: {e}")
        
        # Calculate correlations
        correlation_matrix = self._calculate_correlations(lookback_days)
        summary['correlations'] = correlation_matrix.to_dict() if correlation_matrix is not None else {}
        
        # Generate narrative
        summary['narrative'] = self._generate_narrative(summary)
        
        logger.info(f"Generated market summary for {len(summary['commodities'])} commodities")
        
        return summary
    
    def _calculate_correlations(
        self, 
        lookback_days: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Calculate correlations between commodities.
        
        Parameters
        ----------
        lookback_days : int, optional
            Number of days to look back, by default 30
        
        Returns
        -------
        Optional[pd.DataFrame]
            Correlation matrix
        """
        if not self.commodity_data:
            logger.warning("No commodity data available")
            return None
        
        # Get current date
        current_date = datetime.now()
        
        # Calculate start date
        start_date = current_date - timedelta(days=lookback_days)
        
        # Create DataFrame for prices
        prices = pd.DataFrame()
        
        # Add price data for each commodity
        for commodity, df in self.commodity_data.items():
            try:
                # Filter data for lookback period
                if isinstance(df.index, pd.DatetimeIndex):
                    recent_data = df[df.index >= start_date]
                else:
                    recent_data = df.iloc[-lookback_days:]
                
                if recent_data.empty:
                    continue
                
                # Get price column (assume first column is price)
                price_col = df.columns[0]
                
                # Add to prices DataFrame
                prices[commodity] = recent_data[price_col]
                
            except Exception as e:
                logger.error(f"Error adding {commodity} to correlation matrix: {e}")
        
        # Calculate correlation matrix
        if not prices.empty:
            return prices.corr()
        
        return None
    
    def _generate_narrative(self, summary: Dict[str, Any]) -> str:
        """
        Generate a narrative summary of the market.
        
        Parameters
        ----------
        summary : Dict[str, Any]
            Market summary
        
        Returns
        -------
        str
            Narrative summary
        """
        # Initialize narrative
        narrative = f"Market Summary for {summary['date']} (Past {summary['lookback_days']} Days)\n\n"
        
        # Add commodity summaries
        for commodity, data in summary['commodities'].items():
            # Format commodity name
            commodity_name = commodity.replace('_', ' ').title()
            
            # Format price change
            price_change_str = f"{data['price_change']:.2f} ({data['price_change_pct']:.2%})"
            if data['price_change'] > 0:
                price_change_str = f"+{price_change_str}"
            
            # Add to narrative
            narrative += f"{commodity_name}: ${data['current_price']:.2f}, {price_change_str}\n"
            narrative += f"Trend: {data['trend'].title()}, Volatility: {data['volatility']:.2%}\n\n"
        
        # Add correlation insights
        if summary['correlations']:
            narrative += "Key Correlations:\n"
            
            # Extract correlations
            correlations = pd.DataFrame(summary['correlations'])
            
            # Find strongest positive and negative correlations
            strongest_positive = None
            strongest_negative = None
            max_positive = -1
            max_negative = 1
            
            for i in range(len(correlations.columns)):
                for j in range(i + 1, len(correlations.columns)):
                    commodity1 = correlations.columns[i]
                    commodity2 = correlations.columns[j]
                    corr = correlations.iloc[i, j]
                    
                    if corr > max_positive:
                        max_positive = corr
                        strongest_positive = (commodity1, commodity2, corr)
                    
                    if corr < max_negative:
                        max_negative = corr
                        strongest_negative = (commodity1, commodity2, corr)
            
            # Add to narrative
            if strongest_positive:
                c1, c2, corr = strongest_positive
                narrative += f"- Strongest positive correlation: {c1.replace('_', ' ').title()} and {c2.replace('_', ' ').title()} ({corr:.2f})\n"
            
            if strongest_negative:
                c1, c2, corr = strongest_negative
                narrative += f"- Strongest negative correlation: {c1.replace('_', ' ').title()} and {c2.replace('_', ' ').title()} ({corr:.2f})\n"
            
            narrative += "\n"
        
        # Add overall market sentiment
        # Count trends
        trends = [data['trend'] for data in summary['commodities'].values()]
        up_count = trends.count('up')
        down_count = trends.count('down')
        flat_count = trends.count('flat')
        
        if up_count > down_count + flat_count:
            sentiment = "bullish"
        elif down_count > up_count + flat_count:
            sentiment = "bearish"
        else:
            sentiment = "mixed"
        
        narrative += f"Overall Market Sentiment: {sentiment.title()}\n"
        
        return narrative
    
    def detect_anomalies(
        self, 
        lookback_days: int = 30,
        z_score_threshold: float = 2.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect anomalies in commodity prices.
        
        Parameters
        ----------
        lookback_days : int, optional
            Number of days to look back, by default 30
        z_score_threshold : float, optional
            Z-score threshold for anomalies, by default 2.0
        
        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            Dictionary of anomalies by commodity
        """
        if not self.commodity_data:
            logger.warning("No commodity data available")
            return {}
        
        # Get current date
        current_date = datetime.now()
        
        # Calculate start date
        start_date = current_date - timedelta(days=lookback_days)
        
        # Initialize anomalies
        anomalies = {}
        
        # Detect anomalies for each commodity
        for commodity, df in self.commodity_data.items():
            try:
                # Filter data for lookback period
                if isinstance(df.index, pd.DatetimeIndex):
                    recent_data = df[df.index >= start_date]
                else:
                    recent_data = df.iloc[-lookback_days:]
                
                if recent_data.empty:
                    continue
                
                # Get price column (assume first column is price)
                price_col = df.columns[0]
                
                # Calculate returns
                returns = recent_data[price_col].pct_change().dropna()
                
                # Calculate z-scores
                mean_return = returns.mean()
                std_return = returns.std()
                z_scores = (returns - mean_return) / std_return
                
                # Detect anomalies
                anomaly_dates = z_scores[abs(z_scores) > z_score_threshold].index
                
                if len(anomaly_dates) > 0:
                    # Initialize anomaly list
                    anomalies[commodity] = []
                    
                    # Add anomalies
                    for date in anomaly_dates:
                        return_val = returns.loc[date]
                        z_score = z_scores.loc[date]
                        price = recent_data.loc[date, price_col]
                        
                        anomalies[commodity].append({
                            'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                            'price': price,
                            'return': return_val,
                            'z_score': z_score,
                            'type': 'positive' if z_score > 0 else 'negative'
                        })
                    
                    logger.info(f"Detected {len(anomaly_dates)} anomalies for {commodity}")
                
            except Exception as e:
                logger.error(f"Error detecting anomalies for {commodity}: {e}")
        
        return anomalies
    
    def save_market_summary(
        self, 
        summary: Dict[str, Any],
        file_name: Optional[str] = None
    ) -> str:
        """
        Save market summary to a file.
        
        Parameters
        ----------
        summary : Dict[str, Any]
            Market summary
        file_name : str, optional
            File name, by default None (will generate based on date)
        
        Returns
        -------
        str
            Path to saved file
        """
        # Generate file name if not provided
        if file_name is None:
            date_str = summary.get('date', datetime.now().strftime('%Y-%m-%d'))
            file_name = f"market_summary_{date_str}.md"
        
        # Create file path
        file_path = os.path.join(self.insights_dir, file_name)
        
        # Convert summary to markdown
        markdown = f"# {summary['narrative']}\n\n"
        
        # Add commodity details
        markdown += "## Commodity Details\n\n"
        for commodity, data in summary['commodities'].items():
            commodity_name = commodity.replace('_', ' ').title()
            markdown += f"### {commodity_name}\n\n"
            markdown += f"- Current Price: ${data['current_price']:.2f}\n"
            markdown += f"- Price Change: {data['price_change']:.2f} ({data['price_change_pct']:.2%})\n"
            markdown += f"- Volatility: {data['volatility']:.2%}\n"
            markdown += f"- Trend: {data['trend'].title()}\n"
            markdown += f"- Momentum: {data['momentum']:.2f}\n\n"
        
        # Save to file
        try:
            with open(file_path, 'w') as f:
                f.write(markdown)
            
            logger.info(f"Saved market summary to {file_path}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving market summary: {e}")
            return ""
    
    def save_anomalies(
        self, 
        anomalies: Dict[str, List[Dict[str, Any]]],
        file_name: Optional[str] = None
    ) -> str:
        """
        Save anomalies to a file.
        
        Parameters
        ----------
        anomalies : Dict[str, List[Dict[str, Any]]]
            Dictionary of anomalies by commodity
        file_name : str, optional
            File name, by default None (will generate based on date)
        
        Returns
        -------
        str
            Path to saved file
        """
        # Generate file name if not provided
        if file_name is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
            file_name = f"anomalies_{date_str}.md"
        
        # Create file path
        file_path = os.path.join(self.insights_dir, file_name)
        
        # Convert anomalies to markdown
        markdown = f"# Market Anomalies - {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        if not anomalies:
            markdown += "No anomalies detected.\n"
        else:
            for commodity, anomaly_list in anomalies.items():
                commodity_name = commodity.replace('_', ' ').title()
                markdown += f"## {commodity_name}\n\n"
                
                for anomaly in anomaly_list:
                    anomaly_type = "Positive" if anomaly['type'] == 'positive' else "Negative"
                    markdown += f"### {anomaly_type} Anomaly on {anomaly['date']}\n\n"
                    markdown += f"- Price: ${anomaly['price']:.2f}\n"
                    markdown += f"- Return: {anomaly['return']:.2%}\n"
                    markdown += f"- Z-Score: {anomaly['z_score']:.2f}\n\n"
        
        # Save to file
        try:
            with open(file_path, 'w') as f:
                f.write(markdown)
            
            logger.info(f"Saved anomalies to {file_path}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving anomalies: {e}")
            return ""
    
    def run_daily_analysis(self) -> Dict[str, Any]:
        """
        Run daily market analysis.
        
        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        # Generate market summary
        summary = self.generate_market_summary()
        
        # Detect anomalies
        anomalies = self.detect_anomalies()
        
        # Save results
        summary_path = self.save_market_summary(summary)
        anomalies_path = self.save_anomalies(anomalies)
        
        # Return results
        results = {
            'summary': summary,
            'anomalies': anomalies,
            'summary_path': summary_path,
            'anomalies_path': anomalies_path
        }
        
        return results

def create_market_analyzer() -> MarketAnalyzerAgent:
    """
    Create a market analyzer agent.
    
    Returns
    -------
    MarketAnalyzerAgent
        Market analyzer agent
    """
    return MarketAnalyzerAgent()

def run_market_analysis() -> Dict[str, Any]:
    """
    Run market analysis.
    
    Returns
    -------
    Dict[str, Any]
        Analysis results
    """
    # Create agent
    agent = create_market_analyzer()
    
    # Run analysis
    results = agent.run_daily_analysis()
    
    return results

if __name__ == "__main__":
    # Run market analysis
    results = run_market_analysis()
    
    # Print summary
    print(results['summary']['narrative'])
