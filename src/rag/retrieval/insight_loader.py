"""
Insight loader for the RAG system.
This module loads and processes market insights from markdown files.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/insight_loader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
INSIGHTS_DIR = 'data/insights'

class InsightLoader:
    """
    Class for loading and processing market insights from markdown files.
    """
    
    def __init__(self, insights_dir: str = INSIGHTS_DIR):
        """
        Initialize the insight loader.
        
        Parameters
        ----------
        insights_dir : str, optional
            Directory containing insight files, by default INSIGHTS_DIR
        """
        self.insights_dir = insights_dir
        self.insights = {}
    
    def load_insights(self, commodity: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
        """
        Load insights from markdown files.
        
        Parameters
        ----------
        commodity : str, optional
            Specific commodity to load insights for, by default None (all commodities)
        
        Returns
        -------
        Dict[str, List[Dict[str, str]]]
            Dictionary of insights by commodity
        """
        # Create insights directory if it doesn't exist
        os.makedirs(self.insights_dir, exist_ok=True)
        
        # Get list of insight files
        insight_files = []
        
        if commodity:
            # Look for files matching the specific commodity
            pattern = re.compile(rf"{commodity}.*\.md", re.IGNORECASE)
            insight_files = [f for f in os.listdir(self.insights_dir) if pattern.match(f)]
        else:
            # Get all markdown files
            insight_files = [f for f in os.listdir(self.insights_dir) if f.endswith('.md')]
        
        if not insight_files:
            logger.warning(f"No insight files found in {self.insights_dir}")
            return {}
        
        logger.info(f"Found {len(insight_files)} insight files")
        
        # Process each file
        for file_name in insight_files:
            file_path = os.path.join(self.insights_dir, file_name)
            
            try:
                # Extract commodity name from file name
                commodity_match = re.match(r"([a-zA-Z_]+).*\.md", file_name)
                if not commodity_match:
                    logger.warning(f"Could not extract commodity name from {file_name}")
                    continue
                
                commodity_name = commodity_match.group(1).lower()
                
                # Initialize list for this commodity if not exists
                if commodity_name not in self.insights:
                    self.insights[commodity_name] = []
                
                # Read and process the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse insights from markdown
                insights = self._parse_markdown(content)
                
                # Add to insights dictionary
                self.insights[commodity_name].extend(insights)
                
                logger.info(f"Loaded {len(insights)} insights for {commodity_name} from {file_name}")
                
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
        
        return self.insights
    
    def _parse_markdown(self, content: str) -> List[Dict[str, str]]:
        """
        Parse insights from markdown content.
        
        Parameters
        ----------
        content : str
            Markdown content
        
        Returns
        -------
        List[Dict[str, str]]
            List of insights
        """
        insights = []
        
        # Split by headers
        sections = re.split(r"#{1,6}\s+", content)
        
        # Remove empty sections
        sections = [s.strip() for s in sections if s.strip()]
        
        for section in sections:
            # Extract paragraphs
            paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
            
            for paragraph in paragraphs:
                # Skip bullet points and numbered lists (simplified approach)
                if paragraph.startswith('- ') or paragraph.startswith('* ') or re.match(r"^\d+\.", paragraph):
                    continue
                
                # Add as insight
                insights.append({
                    'content': paragraph,
                    'source': 'market_insights'
                })
        
        return insights
    
    def get_insights_dataframe(self) -> pd.DataFrame:
        """
        Convert insights to a DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame of insights
        """
        rows = []
        
        for commodity, insights in self.insights.items():
            for insight in insights:
                rows.append({
                    'commodity': commodity,
                    'content': insight['content'],
                    'source': insight['source']
                })
        
        return pd.DataFrame(rows)
    
    def save_insights_csv(self, file_path: str) -> None:
        """
        Save insights to a CSV file.
        
        Parameters
        ----------
        file_path : str
            Path to save the CSV file
        """
        df = self.get_insights_dataframe()
        
        if df.empty:
            logger.warning("No insights to save")
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {len(df)} insights to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving insights to CSV: {e}")

def load_and_process_insights(
    insights_dir: str = INSIGHTS_DIR,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Load and process insights from markdown files.
    
    Parameters
    ----------
    insights_dir : str, optional
        Directory containing insight files, by default INSIGHTS_DIR
    output_file : str, optional
        Path to save the CSV file, by default None (don't save)
    
    Returns
    -------
    pd.DataFrame
        DataFrame of insights
    """
    # Create loader
    loader = InsightLoader(insights_dir)
    
    # Load insights
    loader.load_insights()
    
    # Get DataFrame
    df = loader.get_insights_dataframe()
    
    # Save to CSV if output_file provided
    if output_file and not df.empty:
        loader.save_insights_csv(output_file)
    
    return df

if __name__ == "__main__":
    # Example usage
    insights_df = load_and_process_insights(
        output_file='data/processed/insights.csv'
    )
    print(f"Loaded {len(insights_df)} insights")
