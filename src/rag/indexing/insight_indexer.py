"""
Insight indexer for the RAG system.
This module creates a vector index for market insights.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/insight_indexer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
INSIGHTS_CSV = 'data/processed/insights.csv'
CHROMA_DIR = 'data/chroma'

class InsightIndexer:
    """
    Class for indexing market insights using ChromaDB.
    """
    
    def __init__(
        self, 
        insights_csv: str = INSIGHTS_CSV,
        chroma_dir: str = CHROMA_DIR,
        collection_name: str = 'market_insights'
    ):
        """
        Initialize the insight indexer.
        
        Parameters
        ----------
        insights_csv : str, optional
            Path to the CSV file containing insights, by default INSIGHTS_CSV
        chroma_dir : str, optional
            Directory for ChromaDB, by default CHROMA_DIR
        collection_name : str, optional
            Name of the ChromaDB collection, by default 'market_insights'
        """
        self.insights_csv = insights_csv
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.insights_df = None
        self.client = None
        self.collection = None
        self.embedding_function = None
    
    def load_insights(self) -> pd.DataFrame:
        """
        Load insights from CSV file.
        
        Returns
        -------
        pd.DataFrame
            DataFrame of insights
        """
        try:
            if os.path.exists(self.insights_csv):
                self.insights_df = pd.read_csv(self.insights_csv)
                logger.info(f"Loaded {len(self.insights_df)} insights from {self.insights_csv}")
                return self.insights_df
            else:
                logger.warning(f"Insights CSV file not found: {self.insights_csv}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading insights: {e}")
            return pd.DataFrame()
    
    def initialize_chroma(self) -> None:
        """
        Initialize ChromaDB client and collection.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.chroma_dir, exist_ok=True)
            
            # Initialize embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Initialize client
            self.client = chromadb.PersistentClient(path=self.chroma_dir)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Using existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def index_insights(self) -> None:
        """
        Index insights in ChromaDB.
        """
        if self.insights_df is None or self.insights_df.empty:
            logger.warning("No insights to index")
            return
        
        if self.collection is None:
            logger.warning("ChromaDB collection not initialized")
            return
        
        try:
            # Prepare data for indexing
            documents = self.insights_df['content'].tolist()
            ids = [f"insight_{i}" for i in range(len(documents))]
            
            # Add metadata
            metadatas = []
            for _, row in self.insights_df.iterrows():
                metadatas.append({
                    'commodity': row['commodity'],
                    'source': row['source']
                })
            
            # Add to collection
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            
            logger.info(f"Indexed {len(documents)} insights in ChromaDB")
            
        except Exception as e:
            logger.error(f"Error indexing insights: {e}")
            raise
    
    def query_insights(
        self, 
        query: str, 
        n_results: int = 5,
        commodity: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query insights from ChromaDB.
        
        Parameters
        ----------
        query : str
            Query string
        n_results : int, optional
            Number of results to return, by default 5
        commodity : str, optional
            Filter by commodity, by default None (all commodities)
        
        Returns
        -------
        Dict[str, Any]
            Query results
        """
        if self.collection is None:
            logger.warning("ChromaDB collection not initialized")
            return {}
        
        try:
            # Prepare query
            where_clause = None
            if commodity:
                where_clause = {"commodity": commodity}
            
            # Execute query
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause
            )
            
            logger.info(f"Query '{query}' returned {len(results['documents'][0])} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying insights: {e}")
            return {}
    
    def format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format query results.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Query results from ChromaDB
        
        Returns
        -------
        List[Dict[str, Any]]
            Formatted results
        """
        formatted_results = []
        
        if not results or 'documents' not in results or not results['documents']:
            return formatted_results
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        for i, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            formatted_results.append({
                'content': document,
                'commodity': metadata.get('commodity', 'unknown'),
                'source': metadata.get('source', 'unknown'),
                'relevance': 1 - distance  # Convert distance to relevance score
            })
        
        return formatted_results

def create_insight_index(
    insights_csv: str = INSIGHTS_CSV,
    chroma_dir: str = CHROMA_DIR
) -> InsightIndexer:
    """
    Create a vector index for market insights.
    
    Parameters
    ----------
    insights_csv : str, optional
        Path to the CSV file containing insights, by default INSIGHTS_CSV
    chroma_dir : str, optional
        Directory for ChromaDB, by default CHROMA_DIR
    
    Returns
    -------
    InsightIndexer
        Initialized indexer
    """
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create indexer
    indexer = InsightIndexer(insights_csv, chroma_dir)
    
    # Load insights
    indexer.load_insights()
    
    # Initialize ChromaDB
    indexer.initialize_chroma()
    
    # Index insights
    indexer.index_insights()
    
    return indexer

if __name__ == "__main__":
    # Example usage
    indexer = create_insight_index()
    
    # Example query
    results = indexer.query_insights("crude oil price trends")
    formatted_results = indexer.format_results(results)
    
    for i, result in enumerate(formatted_results):
        print(f"Result {i+1} (Relevance: {result['relevance']:.2f}):")
        print(f"Commodity: {result['commodity']}")
        print(f"Content: {result['content']}")
        print()
