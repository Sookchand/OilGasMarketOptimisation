"""
Insight QA agent for the RAG system.
This module implements a question-answering agent using the RAG system.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np

from src.rag.indexing.insight_indexer import InsightIndexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/insight_qa_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
INSIGHTS_CSV = 'data/processed/insights.csv'
CHROMA_DIR = 'data/chroma'

class InsightQAAgent:
    """
    Question-answering agent using the RAG system.
    """
    
    def __init__(
        self, 
        indexer: Optional[InsightIndexer] = None,
        insights_csv: str = INSIGHTS_CSV,
        chroma_dir: str = CHROMA_DIR
    ):
        """
        Initialize the QA agent.
        
        Parameters
        ----------
        indexer : InsightIndexer, optional
            Initialized indexer, by default None (will create a new one)
        insights_csv : str, optional
            Path to the CSV file containing insights, by default INSIGHTS_CSV
        chroma_dir : str, optional
            Directory for ChromaDB, by default CHROMA_DIR
        """
        # Create directories if they don't exist
        os.makedirs('logs', exist_ok=True)
        
        if indexer:
            self.indexer = indexer
        else:
            self.indexer = InsightIndexer(insights_csv, chroma_dir)
            self.indexer.load_insights()
            self.indexer.initialize_chroma()
    
    def answer_question(
        self, 
        question: str, 
        n_results: int = 5,
        commodity: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using the RAG system.
        
        Parameters
        ----------
        question : str
            Question to answer
        n_results : int, optional
            Number of results to retrieve, by default 5
        commodity : str, optional
            Filter by commodity, by default None (all commodities)
        
        Returns
        -------
        Dict[str, Any]
            Answer and supporting information
        """
        logger.info(f"Answering question: {question}")
        
        # Query the index
        results = self.indexer.query_insights(question, n_results, commodity)
        formatted_results = self.indexer.format_results(results)
        
        if not formatted_results:
            logger.warning("No relevant insights found")
            return {
                'answer': "I don't have enough information to answer this question.",
                'sources': []
            }
        
        # Generate answer
        answer = self._generate_answer(question, formatted_results)
        
        # Format response
        response = {
            'answer': answer,
            'sources': formatted_results
        }
        
        return response
    
    def _generate_answer(
        self, 
        question: str, 
        insights: List[Dict[str, Any]]
    ) -> str:
        """
        Generate an answer based on retrieved insights.
        
        Parameters
        ----------
        question : str
            Question to answer
        insights : List[Dict[str, Any]]
            Retrieved insights
        
        Returns
        -------
        str
            Generated answer
        """
        # In a real implementation, this would use an LLM to generate an answer
        # For now, we'll just concatenate the insights with some simple formatting
        
        # Extract content from insights
        contents = [insight['content'] for insight in insights]
        
        # Simple answer generation
        answer = f"Based on the available information:\n\n"
        
        for i, content in enumerate(contents):
            answer += f"{i+1}. {content}\n\n"
        
        return answer

def create_qa_agent(
    insights_csv: str = INSIGHTS_CSV,
    chroma_dir: str = CHROMA_DIR
) -> InsightQAAgent:
    """
    Create a QA agent.
    
    Parameters
    ----------
    insights_csv : str, optional
        Path to the CSV file containing insights, by default INSIGHTS_CSV
    chroma_dir : str, optional
        Directory for ChromaDB, by default CHROMA_DIR
    
    Returns
    -------
    InsightQAAgent
        Initialized QA agent
    """
    # Create indexer
    indexer = InsightIndexer(insights_csv, chroma_dir)
    indexer.load_insights()
    indexer.initialize_chroma()
    
    # Create agent
    agent = InsightQAAgent(indexer)
    
    return agent

if __name__ == "__main__":
    # Example usage
    agent = create_qa_agent()
    
    # Example question
    question = "What are the current trends in crude oil prices?"
    response = agent.answer_question(question)
    
    print(f"Question: {question}")
    print(f"Answer: {response['answer']}")
    print("Sources:")
    for i, source in enumerate(response['sources']):
        print(f"  {i+1}. {source['content'][:100]}... (Relevance: {source['relevance']:.2f})")
