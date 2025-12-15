"""
Vector Store Parameter Experiment

This script evaluates different chunking configurations to find optimal parameters:
1. Chunk size variations (500, 1000, 1500, 2000)
2. Chunk overlap variations
3. TextSplitter type comparison (Recursive, Token, Custom separators)

Usage:
    # Run chunk size experiment
    python evaluate_vectorstore.py --experiment chunk_size --output results/chunk_experiment.json
    
    # Run splitter type experiment  
    python evaluate_vectorstore.py --experiment splitter_type --output results/splitter_experiment.json
    
    # Run full experiment (all combinations)
    python evaluate_vectorstore.py --experiment full --output results/vectorstore_full.json
"""

import json
import os
import sys
import time
import logging
import argparse
import shutil
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict

# LangChain imports
try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        TokenTextSplitter,
        Language,
    )
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available. Install with: pip install langchain-text-splitters langchain-chroma")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ChunkingConfig:
    """Configuration for text chunking"""
    name: str
    chunk_size: int
    chunk_overlap: int
    splitter_type: str = "recursive"  # recursive, token, recursive_custom
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    description: str = ""


@dataclass
class ChunkStats:
    """Statistics about chunks"""
    total_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    std_chunk_size: float
    chunks_per_document: float


@dataclass 
class RetrievalMetrics:
    """Retrieval evaluation metrics"""
    precision_at_k: float
    hit_rate: float
    mrr: float
    avg_relevance_score: float


@dataclass
class ExperimentResult:
    """Result of a single experiment configuration"""
    config_name: str
    config: Dict[str, Any]
    chunk_stats: Dict[str, Any]
    retrieval_metrics: Dict[str, float]
    build_time_seconds: float
    index_size_mb: float


# ============================================================================
# Chunking Configurations
# ============================================================================

# Experiment 1: Chunk Size Variations
CHUNK_SIZE_CONFIGS = [
    ChunkingConfig(
        name="chunk_500",
        chunk_size=500,
        chunk_overlap=100,
        description="Small chunks - fine-grained retrieval, may lose context"
    ),
    ChunkingConfig(
        name="chunk_1000_baseline",
        chunk_size=1000,
        chunk_overlap=200,
        description="Current baseline - balanced"
    ),
    ChunkingConfig(
        name="chunk_1500",
        chunk_size=1500,
        chunk_overlap=300,
        description="Larger chunks - more context, less precise retrieval"
    ),
    ChunkingConfig(
        name="chunk_2000",
        chunk_size=2000,
        chunk_overlap=400,
        description="Maximum context - best for complex topics"
    ),
]

# Experiment 2: Overlap Variations (fixed chunk_size=1000)
OVERLAP_CONFIGS = [
    ChunkingConfig(
        name="overlap_0",
        chunk_size=1000,
        chunk_overlap=0,
        description="No overlap - risk of cutting context"
    ),
    ChunkingConfig(
        name="overlap_100",
        chunk_size=1000,
        chunk_overlap=100,
        description="Minimal overlap"
    ),
    ChunkingConfig(
        name="overlap_200_baseline",
        chunk_size=1000,
        chunk_overlap=200,
        description="Current baseline"
    ),
    ChunkingConfig(
        name="overlap_300",
        chunk_size=1000,
        chunk_overlap=300,
        description="Higher overlap - more redundancy"
    ),
]

# Experiment 3: Splitter Type Variations
SPLITTER_TYPE_CONFIGS = [
    ChunkingConfig(
        name="recursive_default",
        chunk_size=1000,
        chunk_overlap=200,
        splitter_type="recursive",
        separators=["\n\n", "\n", " ", ""],
        description="Default RecursiveCharacterTextSplitter"
    ),
    ChunkingConfig(
        name="recursive_sentence",
        chunk_size=1000,
        chunk_overlap=200,
        splitter_type="recursive",
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        description="Sentence-aware splitting"
    ),
    ChunkingConfig(
        name="token_based",
        chunk_size=256,  # Token count, not characters
        chunk_overlap=50,
        splitter_type="token",
        description="Token-based splitting (256 tokens)"
    ),
    ChunkingConfig(
        name="recursive_paragraph",
        chunk_size=1000,
        chunk_overlap=200,
        splitter_type="recursive",
        separators=["\n\n\n", "\n\n", "\n", " ", ""],
        description="Paragraph-first splitting"
    ),
]


# ============================================================================
# Text Splitter Factory
# ============================================================================

def create_splitter(config: ChunkingConfig):
    """Create a text splitter based on configuration"""
    
    if config.splitter_type == "token":
        return TokenTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    elif config.splitter_type == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )
    elif config.splitter_type == "recursive_python":
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    else:
        raise ValueError(f"Unknown splitter type: {config.splitter_type}")


# ============================================================================
# Document Loading
# ============================================================================

def load_documents_from_ta_system() -> List[Document]:
    """Load documents using TA system's document loaders"""
    try:
        # Try to import from TA system
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from DB import load_documents
        
        documents = load_documents()
        logger.info(f"Loaded {len(documents)} documents from TA system")
        return documents
    except ImportError as e:
        logger.warning(f"Could not import from TA system: {e}")
        return []


def load_sample_documents(source_dir: str = "SOURCE_DOCUMENTS") -> List[Document]:
    """Load sample documents for testing"""
    documents = []
    
    if not os.path.exists(source_dir):
        logger.warning(f"Source directory not found: {source_dir}")
        # Create synthetic documents for testing
        return create_synthetic_documents()
    
    # Simple text file loading
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.txt'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": filepath, "filename": file}
                    ))
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def create_synthetic_documents() -> List[Document]:
    """Create synthetic documents for testing when real data is unavailable"""
    logger.info("Creating synthetic documents for testing")
    
    # Sample content mimicking database course material
    sample_contents = [
        """
        B+ Tree Index Structure
        
        A B+ tree is a self-balancing tree data structure that maintains sorted data 
        and allows searches, sequential access, insertions, and deletions in logarithmic time.
        
        Key Properties:
        - All data is stored in leaf nodes
        - Internal nodes only contain keys for navigation
        - Leaf nodes are linked together for range queries
        - The tree remains balanced through split and merge operations
        
        B+ trees are the most commonly used index structure in database systems because
        they provide excellent performance for both point queries and range queries.
        The fan-out of internal nodes is typically high (100-1000) to minimize tree height.
        """,
        """
        Buffer Pool Management
        
        The buffer pool is a region of main memory used to cache database pages from disk.
        Since disk I/O is orders of magnitude slower than memory access, effective buffer
        management is critical for database performance.
        
        Page Replacement Policies:
        1. LRU (Least Recently Used) - evict the page that hasn't been used longest
        2. Clock - approximation of LRU with lower overhead
        3. LRU-K - considers the K-th most recent access
        
        The buffer pool manager handles:
        - Page fetching from disk
        - Dirty page tracking
        - Pin counting for concurrent access
        - Write-back policies (force vs no-force)
        """,
        """
        Query Optimization
        
        Query optimization is the process of selecting the most efficient execution plan
        for a SQL query. The optimizer considers:
        
        1. Access Methods
           - Sequential scan
           - Index scan
           - Index-only scan
        
        2. Join Algorithms
           - Nested loop join
           - Hash join
           - Sort-merge join
        
        3. Join Ordering
           - Left-deep trees
           - Bushy trees
           - Dynamic programming approach
        
        Cost Estimation:
        The optimizer uses statistics (histograms, cardinality estimates) to estimate
        the cost of different plans. The cost model typically includes I/O cost and CPU cost.
        """,
        """
        Concurrency Control
        
        Concurrency control ensures that concurrent transactions produce correct results.
        
        Two-Phase Locking (2PL):
        - Growing phase: acquire locks
        - Shrinking phase: release locks
        - Guarantees serializability
        
        MVCC (Multi-Version Concurrency Control):
        - Maintains multiple versions of data
        - Readers don't block writers
        - Writers don't block readers
        - Used by PostgreSQL, MySQL InnoDB
        
        Timestamp Ordering:
        - Each transaction gets a timestamp
        - Operations validated based on timestamps
        - Conflicts cause aborts
        """,
        """
        Database Recovery
        
        Recovery ensures durability and atomicity after system failures.
        
        Write-Ahead Logging (WAL):
        - Log records written before data pages
        - Enables redo and undo operations
        
        ARIES Recovery Algorithm:
        1. Analysis Phase
           - Scan log to find dirty pages and active transactions
        
        2. Redo Phase
           - Replay all logged actions
           - Restore database to crash state
        
        3. Undo Phase
           - Rollback uncommitted transactions
           - Process in reverse log order
        
        Checkpointing reduces recovery time by periodically flushing dirty pages.
        """
    ]
    
    documents = []
    for i, content in enumerate(sample_contents):
        documents.append(Document(
            page_content=content.strip(),
            metadata={
                "source": f"synthetic_doc_{i+1}.txt",
                "unit_no": str(i+1).zfill(2),
                "type": "synthetic"
            }
        ))
    
    # Duplicate and vary content to create more documents
    for i, doc in enumerate(documents[:]):
        # Create variations
        documents.append(Document(
            page_content=doc.page_content + "\n\nAdditional details and examples would go here...",
            metadata={
                "source": f"synthetic_doc_{i+1}_extended.txt",
                "unit_no": doc.metadata["unit_no"],
                "type": "synthetic"
            }
        ))
    
    logger.info(f"Created {len(documents)} synthetic documents")
    return documents


# ============================================================================
# Chunk Analysis
# ============================================================================

def analyze_chunks(chunks: List[Document]) -> ChunkStats:
    """Analyze chunk statistics"""
    if not chunks:
        return ChunkStats(0, 0, 0, 0, 0, 0)
    
    sizes = [len(chunk.page_content) for chunk in chunks]
    
    # Count unique source documents
    sources = set(chunk.metadata.get("source", "") for chunk in chunks)
    chunks_per_doc = len(chunks) / len(sources) if sources else 0
    
    mean_size = sum(sizes) / len(sizes)
    variance = sum((s - mean_size) ** 2 for s in sizes) / len(sizes)
    std_size = variance ** 0.5
    
    return ChunkStats(
        total_chunks=len(chunks),
        avg_chunk_size=mean_size,
        min_chunk_size=min(sizes),
        max_chunk_size=max(sizes),
        std_chunk_size=std_size,
        chunks_per_document=chunks_per_doc
    )


# ============================================================================
# Vector Store Building
# ============================================================================

def build_temp_vectorstore(
    chunks: List[Document],
    embeddings,
    persist_dir: str
) -> Chroma:
    """Build a temporary vector store for evaluation"""
    
    # Clean up if exists
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="eval_collection"
    )
    
    return db


# ============================================================================
# Retrieval Evaluation
# ============================================================================

def evaluate_retrieval(
    vectorstore: Chroma,
    test_cases: List[Dict],
    k: int = 4
) -> Tuple[RetrievalMetrics, List[Dict]]:
    """Evaluate retrieval quality on test cases"""
    
    results = []
    precision_scores = []
    hit_scores = []
    mrr_scores = []
    relevance_scores = []
    
    for case in test_cases:
        question = case.get("question", "")
        keywords = case.get("keywords", [])
        expected_units = case.get("relevant_units", [])
        
        # Retrieve documents
        try:
            # Try with relevance scores
            docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
                question, k=k
            )
            docs = [doc for doc, score in docs_with_scores]
            scores = [score for doc, score in docs_with_scores]
            avg_score = sum(scores) / len(scores) if scores else 0
        except Exception:
            # Fallback to regular search
            docs = vectorstore.similarity_search(question, k=k)
            scores = []
            avg_score = 0
        
        # Evaluate relevance
        relevance_flags = []
        first_relevant_rank = None
        
        for idx, doc in enumerate(docs):
            content_lower = doc.page_content.lower()
            metadata = doc.metadata or {}
            
            # Check keyword match
            keyword_match = any(kw.lower() in content_lower for kw in keywords)
            
            # Check unit match
            doc_unit = str(metadata.get("unit_no", "")).zfill(2)
            unit_match = any(str(u).zfill(2) == doc_unit for u in expected_units)
            
            is_relevant = keyword_match or unit_match
            relevance_flags.append(is_relevant)
            
            if is_relevant and first_relevant_rank is None:
                first_relevant_rank = idx + 1
        
        # Calculate metrics
        precision = sum(relevance_flags) / k if k > 0 else 0
        hit = 1.0 if any(relevance_flags) else 0.0
        mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        
        precision_scores.append(precision)
        hit_scores.append(hit)
        mrr_scores.append(mrr)
        relevance_scores.append(avg_score)
        
        results.append({
            "question_id": case.get("id", ""),
            "precision": precision,
            "hit": hit,
            "mrr": mrr,
            "avg_relevance_score": avg_score,
            "retrieved_chunks": len(docs)
        })
    
    n = len(test_cases) if test_cases else 1
    metrics = RetrievalMetrics(
        precision_at_k=sum(precision_scores) / n,
        hit_rate=sum(hit_scores) / n,
        mrr=sum(mrr_scores) / n,
        avg_relevance_score=sum(relevance_scores) / n
    )
    
    return metrics, results


# ============================================================================
# Main Experiment Runner
# ============================================================================

class VectorStoreExperiment:
    """Main experiment runner for vector store evaluation"""
    
    def __init__(
        self,
        test_dataset_path: str,
        embedding_model: str = None,
        device_type: str = "cuda"
    ):
        """
        Initialize experiment
        
        Args:
            test_dataset_path: Path to test_dataset.json
            embedding_model: Embedding model name (uses TA system default if None)
            device_type: Device for embeddings
        """
        # Load test dataset
        with open(test_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.test_cases = data['test_cases']
        
        # Load embeddings
        self.embeddings = self._load_embeddings(embedding_model, device_type)
        
        # Load documents
        self.documents = self._load_documents()
        
        # Create temp directory for experiment DBs
        self.temp_dir = tempfile.mkdtemp(prefix="vectorstore_exp_")
        logger.info(f"Using temp directory: {self.temp_dir}")
    
    def _load_embeddings(self, model_name: str, device_type: str):
        """Load embedding model"""
        try:
            # Try TA system embeddings first
            from utils import get_embeddings
            embeddings = get_embeddings(device_type)
            logger.info("Loaded embeddings from TA system")
            return embeddings
        except ImportError:
            pass
        
        # Fallback to HuggingFace embeddings
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device_type if device_type != "cuda" else "cpu"}
            )
            logger.info(f"Loaded HuggingFace embeddings: {model_name}")
            return embeddings
        except ImportError:
            raise ImportError(
                "No embedding model available. Install: pip install langchain-huggingface"
            )
    
    def _load_documents(self) -> List[Document]:
        """Load source documents"""
        # Try TA system first
        docs = load_documents_from_ta_system()
        if docs:
            return docs
        
        # Try local directory
        docs = load_sample_documents("SOURCE_DOCUMENTS")
        if docs:
            return docs
        
        # Use synthetic documents
        return create_synthetic_documents()
    
    def run_single_config(self, config: ChunkingConfig) -> ExperimentResult:
        """Run experiment for a single configuration"""
        logger.info(f"Testing configuration: {config.name}")
        
        start_time = time.time()
        
        # Create splitter and chunk documents
        splitter = create_splitter(config)
        chunks = splitter.split_documents(self.documents)
        
        # Analyze chunks
        chunk_stats = analyze_chunks(chunks)
        logger.info(f"  Created {chunk_stats.total_chunks} chunks, avg size: {chunk_stats.avg_chunk_size:.0f}")
        
        # Build vector store
        persist_dir = os.path.join(self.temp_dir, config.name)
        vectorstore = build_temp_vectorstore(chunks, self.embeddings, persist_dir)
        
        build_time = time.time() - start_time
        
        # Get index size
        index_size_mb = 0
        if os.path.exists(persist_dir):
            for root, dirs, files in os.walk(persist_dir):
                for f in files:
                    index_size_mb += os.path.getsize(os.path.join(root, f))
            index_size_mb /= (1024 * 1024)
        
        # Evaluate retrieval
        metrics, detailed_results = evaluate_retrieval(
            vectorstore, 
            self.test_cases,
            k=4
        )
        
        logger.info(f"  Precision@4: {metrics.precision_at_k:.3f}, Hit Rate: {metrics.hit_rate:.3f}")
        
        return ExperimentResult(
            config_name=config.name,
            config=asdict(config),
            chunk_stats=asdict(chunk_stats),
            retrieval_metrics={
                "precision_at_k": metrics.precision_at_k,
                "hit_rate": metrics.hit_rate,
                "mrr": metrics.mrr,
                "avg_relevance_score": metrics.avg_relevance_score
            },
            build_time_seconds=build_time,
            index_size_mb=index_size_mb
        )
    
    def run_experiment(
        self,
        experiment_type: str = "chunk_size"
    ) -> Dict[str, Any]:
        """
        Run a complete experiment
        
        Args:
            experiment_type: One of "chunk_size", "overlap", "splitter_type", "full"
        """
        # Select configurations
        if experiment_type == "chunk_size":
            configs = CHUNK_SIZE_CONFIGS
        elif experiment_type == "overlap":
            configs = OVERLAP_CONFIGS
        elif experiment_type == "splitter_type":
            configs = SPLITTER_TYPE_CONFIGS
        elif experiment_type == "full":
            configs = CHUNK_SIZE_CONFIGS + OVERLAP_CONFIGS + SPLITTER_TYPE_CONFIGS
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        logger.info(f"Starting {experiment_type} experiment with {len(configs)} configurations")
        
        results = []
        for config in configs:
            result = self.run_single_config(config)
            results.append(asdict(result))
        
        # Find best configuration
        best_by_precision = max(results, key=lambda x: x["retrieval_metrics"]["precision_at_k"])
        best_by_hit_rate = max(results, key=lambda x: x["retrieval_metrics"]["hit_rate"])
        
        output = {
            "experiment_type": experiment_type,
            "timestamp": datetime.now().isoformat(),
            "num_documents": len(self.documents),
            "num_test_cases": len(self.test_cases),
            "results": results,
            "summary": {
                "best_by_precision": best_by_precision["config_name"],
                "best_precision_score": best_by_precision["retrieval_metrics"]["precision_at_k"],
                "best_by_hit_rate": best_by_hit_rate["config_name"],
                "best_hit_rate_score": best_by_hit_rate["retrieval_metrics"]["hit_rate"],
            }
        }
        
        return output
    
    def cleanup(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")


# ============================================================================
# Results Analysis and Visualization
# ============================================================================

def analyze_vectorstore_results(results: Dict[str, Any], output_dir: str):
    """Analyze and visualize vectorstore experiment results"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate markdown report
    report_lines = [
        "# Vector Store Parameter Experiment Report",
        "",
        f"**Experiment Type:** {results.get('experiment_type', 'N/A')}",
        f"**Timestamp:** {results.get('timestamp', 'N/A')}",
        f"**Documents:** {results.get('num_documents', 'N/A')}",
        f"**Test Cases:** {results.get('num_test_cases', 'N/A')}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"**Best by Precision@K:** {results['summary']['best_by_precision']} ({results['summary']['best_precision_score']:.3f})",
        f"**Best by Hit Rate:** {results['summary']['best_by_hit_rate']} ({results['summary']['best_hit_rate_score']:.3f})",
        "",
        "---",
        "",
        "## Detailed Results",
        "",
        "| Configuration | Chunk Size | Overlap | Chunks | Precision@K | Hit Rate | MRR | Build Time (s) |",
        "|--------------|------------|---------|--------|-------------|----------|-----|----------------|",
    ]
    
    for r in results['results']:
        config = r['config']
        metrics = r['retrieval_metrics']
        stats = r['chunk_stats']
        report_lines.append(
            f"| {r['config_name']} | {config['chunk_size']} | {config['chunk_overlap']} | "
            f"{stats['total_chunks']} | {metrics['precision_at_k']:.3f} | "
            f"{metrics['hit_rate']:.3f} | {metrics['mrr']:.3f} | {r['build_time_seconds']:.1f} |"
        )
    
    report_lines.extend([
        "",
        "---",
        "",
        "## Chunk Statistics",
        "",
        "| Configuration | Total Chunks | Avg Size | Min Size | Max Size | Std Dev |",
        "|--------------|--------------|----------|----------|----------|---------|",
    ])
    
    for r in results['results']:
        stats = r['chunk_stats']
        report_lines.append(
            f"| {r['config_name']} | {stats['total_chunks']} | {stats['avg_chunk_size']:.0f} | "
            f"{stats['min_chunk_size']} | {stats['max_chunk_size']} | {stats['std_chunk_size']:.0f} |"
        )
    
    report_lines.extend([
        "",
        "---",
        "",
        "## Recommendations",
        "",
    ])
    
    # Add recommendations based on results
    best_config = results['summary']['best_by_precision']
    report_lines.append(f"Based on the experiment results, **{best_config}** provides the best retrieval precision.")
    
    with open(os.path.join(output_dir, "vectorstore_report.md"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Generate visualization
    try:
        import matplotlib.pyplot as plt
        
        configs = [r['config_name'] for r in results['results']]
        precisions = [r['retrieval_metrics']['precision_at_k'] for r in results['results']]
        hit_rates = [r['retrieval_metrics']['hit_rate'] for r in results['results']]
        mrrs = [r['retrieval_metrics']['mrr'] for r in results['results']]
        chunk_counts = [r['chunk_stats']['total_chunks'] for r in results['results']]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Chart 1: Retrieval Metrics Comparison
        x = range(len(configs))
        width = 0.25
        
        axes[0, 0].bar([i - width for i in x], precisions, width, label='Precision@K', color='#2ecc71')
        axes[0, 0].bar(x, hit_rates, width, label='Hit Rate', color='#3498db')
        axes[0, 0].bar([i + width for i in x], mrrs, width, label='MRR', color='#9b59b6')
        axes[0, 0].set_xlabel('Configuration')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Retrieval Performance by Configuration')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(configs, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1.1)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Chart 2: Chunk Count vs Performance
        axes[0, 1].scatter(chunk_counts, precisions, s=100, c='#e74c3c', label='Precision@K')
        axes[0, 1].scatter(chunk_counts, hit_rates, s=100, c='#3498db', marker='^', label='Hit Rate')
        for i, config in enumerate(configs):
            axes[0, 1].annotate(config, (chunk_counts[i], precisions[i]), fontsize=8)
        axes[0, 1].set_xlabel('Total Chunks')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Chunk Count vs Retrieval Performance')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Chart 3: Chunk Size Distribution
        chunk_sizes = [r['config']['chunk_size'] for r in results['results']]
        axes[1, 0].bar(configs, chunk_sizes, color='#f39c12')
        axes[1, 0].set_xlabel('Configuration')
        axes[1, 0].set_ylabel('Chunk Size')
        axes[1, 0].set_title('Chunk Size by Configuration')
        axes[1, 0].set_xticklabels(configs, rotation=45, ha='right')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Chart 4: Build Time and Index Size
        build_times = [r['build_time_seconds'] for r in results['results']]
        index_sizes = [r['index_size_mb'] for r in results['results']]
        
        ax2 = axes[1, 1].twinx()
        bars1 = axes[1, 1].bar([i - 0.2 for i in x], build_times, 0.4, label='Build Time (s)', color='#1abc9c')
        bars2 = ax2.bar([i + 0.2 for i in x], index_sizes, 0.4, label='Index Size (MB)', color='#e67e22')
        axes[1, 1].set_xlabel('Configuration')
        axes[1, 1].set_ylabel('Build Time (seconds)')
        ax2.set_ylabel('Index Size (MB)')
        axes[1, 1].set_title('Build Time and Index Size')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(configs, rotation=45, ha='right')
        axes[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vectorstore_experiment.png'), dpi=150)
        plt.close()
        
        logger.info(f"Saved visualization to {output_dir}/vectorstore_experiment.png")
        
    except ImportError:
        logger.warning("matplotlib not available. Skipping visualization.")
    
    # Save CSV
    with open(os.path.join(output_dir, 'vectorstore_results.csv'), 'w', encoding='utf-8') as f:
        f.write("config_name,chunk_size,chunk_overlap,splitter_type,total_chunks,avg_chunk_size,")
        f.write("precision_at_k,hit_rate,mrr,build_time_seconds,index_size_mb\n")
        for r in results['results']:
            config = r['config']
            stats = r['chunk_stats']
            metrics = r['retrieval_metrics']
            f.write(f"{r['config_name']},{config['chunk_size']},{config['chunk_overlap']},")
            f.write(f"{config['splitter_type']},{stats['total_chunks']},{stats['avg_chunk_size']:.0f},")
            f.write(f"{metrics['precision_at_k']:.4f},{metrics['hit_rate']:.4f},{metrics['mrr']:.4f},")
            f.write(f"{r['build_time_seconds']:.2f},{r['index_size_mb']:.2f}\n")
    
    logger.info(f"Analysis saved to {output_dir}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Vector Store Parameter Experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        default="test_dataset.json",
        help="Path to test dataset JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vectorstore_experiment.json",
        help="Path to save experiment results"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["chunk_size", "overlap", "splitter_type", "full"],
        default="chunk_size",
        help="Type of experiment to run"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for embeddings (cpu, cuda)"
    )
    parser.add_argument(
        "--analyze-only",
        type=str,
        default=None,
        help="Only analyze existing results file (skip experiment)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vectorstore_analysis",
        help="Directory for analysis outputs"
    )
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Just analyze existing results
        with open(args.analyze_only, 'r', encoding='utf-8') as f:
            results = json.load(f)
        analyze_vectorstore_results(results, args.output_dir)
        return
    
    # Run experiment
    experiment = VectorStoreExperiment(
        test_dataset_path=args.dataset,
        device_type=args.device
    )
    
    try:
        results = experiment.run_experiment(experiment_type=args.experiment)
        
        # Save results
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {args.output}")
        
        # Analyze results
        analyze_vectorstore_results(results, args.output_dir)
        
    finally:
        experiment.cleanup()


if __name__ == "__main__":
    main()
