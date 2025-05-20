"""
Hybrid retrieval system for the RAG component.
This module provides a hybrid retrieval system that combines semantic and keyword search.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hybrid_retriever.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class Document:
    """
    Document class for storing document information.
    """

    def __init__(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a document.

        Parameters
        ----------
        doc_id : str
            Document ID
        text : str
            Document text
        metadata : Dict[str, Any], optional
            Document metadata, by default None
        """
        self.doc_id = doc_id
        self.text = text
        self.metadata = metadata or {}

    def __str__(self) -> str:
        return f"Document(id={self.doc_id}, text={self.text[:50]}...)"

    def __repr__(self) -> str:
        return self.__str__()

class KeywordIndex:
    """
    Keyword index for efficient keyword search.
    """

    def __init__(self, use_bm25: bool = True):
        """
        Initialize the keyword index.

        Parameters
        ----------
        use_bm25 : bool, optional
            Whether to use BM25 for ranking, by default True
        """
        self.documents = []
        self.doc_texts = []
        self.use_bm25 = use_bm25
        self.bm25 = None
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_df=0.85,
            min_df=2,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the index.

        Parameters
        ----------
        documents : List[Document]
            List of documents to add
        """
        self.documents.extend(documents)
        self.doc_texts = [doc.text for doc in self.documents]

        if self.use_bm25:
            # Tokenize documents for BM25
            tokenized_docs = [word_tokenize(doc.lower()) for doc in self.doc_texts]
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            # Build TF-IDF matrix
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.doc_texts)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for documents matching the query.

        Parameters
        ----------
        query : str
            Search query
        top_k : int, optional
            Number of top results to return, by default 5

        Returns
        -------
        List[Tuple[Document, float]]
            List of (document, score) tuples
        """
        if not self.documents:
            return []

        if self.use_bm25:
            # Tokenize query
            tokenized_query = word_tokenize(query.lower())

            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
        else:
            # Transform query to TF-IDF vector
            query_vector = self.tfidf_vectorizer.transform([query])

            # Calculate cosine similarity
            scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        # Get top-k documents
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Return documents and scores
        results = [(self.documents[i], scores[i]) for i in top_indices if scores[i] > 0]

        return results

class VectorIndex:
    """
    Vector index for efficient semantic search.
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        dimension: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the vector index.

        Parameters
        ----------
        model_name : str, optional
            Name of the sentence transformer model, by default 'all-MiniLM-L6-v2'
        dimension : int, optional
            Dimension of the vectors, by default None (will be inferred from the model)
        device : str, optional
            Device to use for the model, by default None (will use GPU if available)
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)

        # Set dimension
        if dimension is None:
            self.dimension = self.model.get_sentence_embedding_dimension()
        else:
            self.dimension = dimension

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)

        # Store documents
        self.documents = []

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the index.

        Parameters
        ----------
        documents : List[Document]
            List of documents to add
        """
        if not documents:
            return

        # Store documents
        start_idx = len(self.documents)
        self.documents.extend(documents)

        # Get document texts
        texts = [doc.text for doc in documents]

        # Encode texts
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        # Add to index
        self.index.add(np.array(embeddings).astype('float32'))

        logger.info(f"Added {len(documents)} documents to vector index")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for documents matching the query.

        Parameters
        ----------
        query : str
            Search query
        top_k : int, optional
            Number of top results to return, by default 5

        Returns
        -------
        List[Tuple[Document, float]]
            List of (document, score) tuples
        """
        if not self.documents:
            return []

        # Encode query
        query_embedding = self.model.encode(query, convert_to_numpy=True)

        # Search index
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'),
            min(top_k, len(self.documents))
        )

        # Return documents and scores
        results = [(self.documents[int(i)], float(1.0 / (1.0 + distances[0][j])))
                  for j, i in enumerate(indices[0])]

        return results

class Reranker:
    """
    Reranker for improving search results.
    """

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize the reranker.

        Parameters
        ----------
        model_name : str, optional
            Name of the cross-encoder model, by default 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        """
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except ImportError:
            logger.warning("CrossEncoder not available. Reranker will not be used.")
            self.model = None

    def rerank(
        self,
        query: str,
        results: List[Tuple[Document, float]],
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Rerank search results.

        Parameters
        ----------
        query : str
            Search query
        results : List[Tuple[Document, float]]
            List of (document, score) tuples
        top_k : int, optional
            Number of top results to return, by default 5

        Returns
        -------
        List[Tuple[Document, float]]
            Reranked list of (document, score) tuples
        """
        if not results or self.model is None:
            return results

        # Prepare input for reranker
        pairs = [(query, doc.text) for doc, _ in results]

        # Get scores
        scores = self.model.predict(pairs)

        # Sort by score
        reranked_results = [(results[i][0], float(scores[i]))
                           for i in np.argsort(scores)[::-1][:top_k]]

        return reranked_results

class HybridRetriever:
    """
    Hybrid retriever that combines semantic and keyword search.
    """

    def __init__(
        self,
        vector_index: Optional[VectorIndex] = None,
        keyword_index: Optional[KeywordIndex] = None,
        reranker: Optional[Reranker] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """
        Initialize the hybrid retriever.

        Parameters
        ----------
        vector_index : VectorIndex, optional
            Vector index for semantic search, by default None (will create a new one)
        keyword_index : KeywordIndex, optional
            Keyword index for keyword search, by default None (will create a new one)
        reranker : Reranker, optional
            Reranker for improving search results, by default None (will create a new one)
        vector_weight : float, optional
            Weight for vector search results, by default 0.7
        keyword_weight : float, optional
            Weight for keyword search results, by default 0.3
        """
        self.vector_index = vector_index or VectorIndex()
        self.keyword_index = keyword_index or KeywordIndex()
        self.reranker = reranker or Reranker()
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the retriever.

        Parameters
        ----------
        documents : List[Document]
            List of documents to add
        """
        self.vector_index.add_documents(documents)
        self.keyword_index.add_documents(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_k: int = 5,
        use_reranker: bool = True
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents matching the query.

        Parameters
        ----------
        query : str
            Search query
        top_k : int, optional
            Number of top results to return, by default 10
        rerank_top_k : int, optional
            Number of top results to rerank, by default 5
        use_reranker : bool, optional
            Whether to use the reranker, by default True

        Returns
        -------
        List[Tuple[Document, float]]
            List of (document, score) tuples
        """
        # Get semantic search results
        vector_results = self.vector_index.search(query, top_k=top_k)

        # Get keyword search results
        keyword_results = self.keyword_index.search(query, top_k=top_k)

        # Combine results with deduplication
        combined_results = self._merge_results(
            vector_results,
            keyword_results,
            self.vector_weight,
            self.keyword_weight
        )

        # Rerank if requested
        if use_reranker and self.reranker and len(combined_results) > rerank_top_k:
            combined_results = self.reranker.rerank(query, combined_results, top_k=rerank_top_k)

        return combined_results

    def _merge_results(
        self,
        vector_results: List[Tuple[Document, float]],
        keyword_results: List[Tuple[Document, float]],
        vector_weight: float,
        keyword_weight: float
    ) -> List[Tuple[Document, float]]:
        """
        Merge results from vector and keyword search.

        Parameters
        ----------
        vector_results : List[Tuple[Document, float]]
            Results from vector search
        keyword_results : List[Tuple[Document, float]]
            Results from keyword search
        vector_weight : float
            Weight for vector search results
        keyword_weight : float
            Weight for keyword search results

        Returns
        -------
        List[Tuple[Document, float]]
            Merged results
        """
        # Create a dictionary to store merged results
        merged = {}

        # Add vector results
        for doc, score in vector_results:
            merged[doc.doc_id] = {
                'document': doc,
                'vector_score': score,
                'keyword_score': 0.0
            }

        # Add keyword results
        for doc, score in keyword_results:
            if doc.doc_id in merged:
                merged[doc.doc_id]['keyword_score'] = score
            else:
                merged[doc.doc_id] = {
                    'document': doc,
                    'vector_score': 0.0,
                    'keyword_score': score
                }

        # Calculate combined scores
        for doc_id, data in merged.items():
            data['combined_score'] = (
                vector_weight * data['vector_score'] +
                keyword_weight * data['keyword_score']
            )

        # Sort by combined score
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )

        # Return documents and combined scores
        return [(data['document'], data['combined_score']) for data in sorted_results]
