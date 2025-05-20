"""
Script to demonstrate the usage of the hybrid retriever.
This script loads documents, indexes them, and performs hybrid retrieval.
"""

import os
import sys
import logging
import argparse
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import uuid

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.rag.retrieval.hybrid_retriever import Document, HybridRetriever, VectorIndex, KeywordIndex, Reranker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hybrid_retrieval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_documents_from_csv(
    file_path: str,
    text_column: str,
    id_column: Optional[str] = None,
    metadata_columns: Optional[List[str]] = None
) -> List[Document]:
    """
    Load documents from a CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file
    text_column : str
        Name of the column containing the document text
    id_column : str, optional
        Name of the column containing the document ID, by default None
    metadata_columns : List[str], optional
        List of column names to include as metadata, by default None
    
    Returns
    -------
    List[Document]
        List of documents
    """
    try:
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Check if text column exists
        if text_column not in df.columns:
            logger.error(f"Text column '{text_column}' not found in CSV file")
            return []
        
        # Create documents
        documents = []
        
        for _, row in df.iterrows():
            # Get document ID
            if id_column and id_column in df.columns:
                doc_id = str(row[id_column])
            else:
                doc_id = str(uuid.uuid4())
            
            # Get document text
            text = str(row[text_column])
            
            # Get metadata
            metadata = {}
            if metadata_columns:
                for col in metadata_columns:
                    if col in df.columns:
                        metadata[col] = row[col]
            
            # Create document
            doc = Document(doc_id=doc_id, text=text, metadata=metadata)
            documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    
    except Exception as e:
        logger.error(f"Error loading documents from CSV file: {e}")
        return []

def load_documents_from_json(
    file_path: str,
    text_field: str,
    id_field: Optional[str] = None,
    metadata_fields: Optional[List[str]] = None
) -> List[Document]:
    """
    Load documents from a JSON file.
    
    Parameters
    ----------
    file_path : str
        Path to the JSON file
    text_field : str
        Name of the field containing the document text
    id_field : str, optional
        Name of the field containing the document ID, by default None
    metadata_fields : List[str], optional
        List of field names to include as metadata, by default None
    
    Returns
    -------
    List[Document]
        List of documents
    """
    try:
        # Load JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if data is a list
        if not isinstance(data, list):
            logger.error("JSON file must contain a list of documents")
            return []
        
        # Create documents
        documents = []
        
        for item in data:
            # Check if text field exists
            if text_field not in item:
                logger.warning(f"Text field '{text_field}' not found in document, skipping")
                continue
            
            # Get document ID
            if id_field and id_field in item:
                doc_id = str(item[id_field])
            else:
                doc_id = str(uuid.uuid4())
            
            # Get document text
            text = str(item[text_field])
            
            # Get metadata
            metadata = {}
            if metadata_fields:
                for field in metadata_fields:
                    if field in item:
                        metadata[field] = item[field]
            
            # Create document
            doc = Document(doc_id=doc_id, text=text, metadata=metadata)
            documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    
    except Exception as e:
        logger.error(f"Error loading documents from JSON file: {e}")
        return []

def create_sample_documents() -> List[Document]:
    """
    Create sample documents for demonstration.
    
    Returns
    -------
    List[Document]
        List of sample documents
    """
    documents = [
        Document(
            doc_id="1",
            text="Crude oil prices rose sharply today due to OPEC production cuts and geopolitical tensions in the Middle East.",
            metadata={"source": "news", "date": "2023-01-15", "topic": "crude_oil"}
        ),
        Document(
            doc_id="2",
            text="Natural gas futures fell as mild weather forecasts reduced heating demand expectations for the coming weeks.",
            metadata={"source": "news", "date": "2023-01-16", "topic": "natural_gas"}
        ),
        Document(
            doc_id="3",
            text="Gasoline prices at the pump increased following the rise in crude oil prices and refinery outages on the Gulf Coast.",
            metadata={"source": "news", "date": "2023-01-17", "topic": "gasoline"}
        ),
        Document(
            doc_id="4",
            text="OPEC+ agreed to maintain current production levels, surprising analysts who expected further cuts to support prices.",
            metadata={"source": "news", "date": "2023-01-18", "topic": "crude_oil"}
        ),
        Document(
            doc_id="5",
            text="Diesel demand is expected to increase as industrial activity picks up in China following the Lunar New Year holiday.",
            metadata={"source": "news", "date": "2023-01-19", "topic": "diesel"}
        ),
        Document(
            doc_id="6",
            text="The EIA reported a larger-than-expected draw in crude oil inventories, indicating strong demand in the US market.",
            metadata={"source": "report", "date": "2023-01-20", "topic": "crude_oil"}
        ),
        Document(
            doc_id="7",
            text="Renewable energy investments reached a record high last quarter as companies accelerate their transition away from fossil fuels.",
            metadata={"source": "report", "date": "2023-01-21", "topic": "renewables"}
        ),
        Document(
            doc_id="8",
            text="Refinery utilization rates increased to 92% as facilities returned from seasonal maintenance ahead of the summer driving season.",
            metadata={"source": "report", "date": "2023-01-22", "topic": "refining"}
        ),
        Document(
            doc_id="9",
            text="Oil traders are closely monitoring tensions between Russia and Ukraine, which could disrupt global energy supplies.",
            metadata={"source": "news", "date": "2023-01-23", "topic": "geopolitics"}
        ),
        Document(
            doc_id="10",
            text="The International Energy Agency revised its oil demand forecast upward, citing stronger-than-expected economic growth in emerging markets.",
            metadata={"source": "report", "date": "2023-01-24", "topic": "demand"}
        )
    ]
    
    logger.info(f"Created {len(documents)} sample documents")
    return documents

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run hybrid retrieval')
    
    parser.add_argument('--query', type=str, default="OPEC oil production cuts",
                        help='Search query')
    
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top results to return')
    
    parser.add_argument('--csv-file', type=str, default=None,
                        help='Path to CSV file containing documents')
    
    parser.add_argument('--json-file', type=str, default=None,
                        help='Path to JSON file containing documents')
    
    parser.add_argument('--text-column', type=str, default='text',
                        help='Name of the column/field containing the document text')
    
    parser.add_argument('--id-column', type=str, default=None,
                        help='Name of the column/field containing the document ID')
    
    parser.add_argument('--metadata-columns', nargs='+', default=None,
                        help='List of column/field names to include as metadata')
    
    parser.add_argument('--vector-weight', type=float, default=0.7,
                        help='Weight for vector search results')
    
    parser.add_argument('--keyword-weight', type=float, default=0.3,
                        help='Weight for keyword search results')
    
    parser.add_argument('--use-reranker', action='store_true',
                        help='Whether to use the reranker')
    
    return parser.parse_args()

def main():
    """Run hybrid retrieval."""
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    
    # Load documents
    documents = []
    
    if args.csv_file:
        documents = load_documents_from_csv(
            file_path=args.csv_file,
            text_column=args.text_column,
            id_column=args.id_column,
            metadata_columns=args.metadata_columns
        )
    elif args.json_file:
        documents = load_documents_from_json(
            file_path=args.json_file,
            text_field=args.text_column,
            id_field=args.id_column,
            metadata_fields=args.metadata_columns
        )
    else:
        # Use sample documents
        documents = create_sample_documents()
    
    if not documents:
        logger.error("No documents loaded")
        return
    
    # Initialize hybrid retriever
    retriever = HybridRetriever(
        vector_weight=args.vector_weight,
        keyword_weight=args.keyword_weight
    )
    
    # Add documents to retriever
    retriever.add_documents(documents)
    
    # Perform retrieval
    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        use_reranker=args.use_reranker
    )
    
    # Print results
    print(f"\nQuery: {args.query}")
    print(f"Top {len(results)} results:")
    print("=" * 80)
    
    for i, (doc, score) in enumerate(results):
        print(f"{i+1}. Score: {score:.4f}")
        print(f"   ID: {doc.doc_id}")
        print(f"   Text: {doc.text}")
        if doc.metadata:
            print(f"   Metadata: {doc.metadata}")
        print("-" * 80)

if __name__ == "__main__":
    main()
