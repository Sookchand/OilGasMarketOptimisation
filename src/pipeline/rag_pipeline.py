"""
RAG pipeline for the Oil & Gas Market Optimization project.
This script orchestrates the RAG system pipeline.
"""

import os
import logging
import argparse
from typing import Dict, List, Optional, Union, Any, Tuple

from src.rag.retrieval.insight_loader import load_and_process_insights
from src.rag.indexing.insight_indexer import create_insight_index
from src.rag.agents.insight_qa_agent import create_qa_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
INSIGHTS_DIR = 'data/insights'
PROCESSED_DIR = 'data/processed'
CHROMA_DIR = 'data/chroma'

def run_rag_pipeline(
    insights_dir: str = INSIGHTS_DIR,
    processed_dir: str = PROCESSED_DIR,
    chroma_dir: str = CHROMA_DIR,
    steps: Optional[List[str]] = None
) -> None:
    """
    Run the RAG pipeline.
    
    Parameters
    ----------
    insights_dir : str, optional
        Directory containing insight files, by default INSIGHTS_DIR
    processed_dir : str, optional
        Directory for processed data, by default PROCESSED_DIR
    chroma_dir : str, optional
        Directory for ChromaDB, by default CHROMA_DIR
    steps : List[str], optional
        List of pipeline steps to run, by default None (all steps)
    """
    # Create directories if they don't exist
    os.makedirs(insights_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(chroma_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Default steps if not specified
    if steps is None:
        steps = ['load', 'index', 'qa']
    
    logger.info(f"Running RAG pipeline with steps: {steps}")
    
    # Step 1: Load and process insights
    insights_csv = os.path.join(processed_dir, 'insights.csv')
    
    if 'load' in steps:
        logger.info("Step 1: Loading and processing insights")
        insights_df = load_and_process_insights(insights_dir, insights_csv)
        logger.info(f"Processed {len(insights_df)} insights")
    
    # Step 2: Create vector index
    if 'index' in steps:
        logger.info("Step 2: Creating vector index")
        indexer = create_insight_index(insights_csv, chroma_dir)
        logger.info("Vector index created")
    
    # Step 3: Create QA agent
    if 'qa' in steps:
        logger.info("Step 3: Creating QA agent")
        agent = create_qa_agent(insights_csv, chroma_dir)
        logger.info("QA agent created")
        
        # Example question
        question = "What are the current trends in crude oil prices?"
        logger.info(f"Example question: {question}")
        
        response = agent.answer_question(question)
        logger.info(f"Answer: {response['answer'][:100]}...")
    
    logger.info("RAG pipeline completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the RAG pipeline')
    
    parser.add_argument('--insights-dir', default=INSIGHTS_DIR, help='Directory containing insight files')
    parser.add_argument('--processed-dir', default=PROCESSED_DIR, help='Directory for processed data')
    parser.add_argument('--chroma-dir', default=CHROMA_DIR, help='Directory for ChromaDB')
    parser.add_argument('--steps', nargs='+', choices=['load', 'index', 'qa'],
                        help='Pipeline steps to run')
    
    args = parser.parse_args()
    
    run_rag_pipeline(
        insights_dir=args.insights_dir,
        processed_dir=args.processed_dir,
        chroma_dir=args.chroma_dir,
        steps=args.steps
    )
