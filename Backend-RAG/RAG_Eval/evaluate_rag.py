"""
RAG System Evaluation Script for TA Assistant

This script evaluates the TA Assistant RAG system using:
1. Retrieval metrics (Precision@K, Hit Rate, MRR)
2. Generation metrics (Correctness, Relevance, Faithfulness via LLM-as-Judge)
3. Ablation study across different configurations

Usage:
    python evaluate_rag.py --config baseline --output results/baseline.json
    python evaluate_rag.py --config full --output results/full.json
    python evaluate_rag.py --ablation --output results/ablation_study.json
"""

import json
import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# LLM-as-Judge API (외부 API 사용)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RetrievalResult:
    """Single retrieval evaluation result"""
    question_id: str
    precision_at_k: float
    hit_rate: float
    mrr: float
    retrieved_units: List[str]
    expected_units: List[str]
    keyword_matches: int
    total_keywords: int


@dataclass
class GenerationResult:
    """Single generation evaluation result"""
    question_id: str
    correctness: float
    relevance: float
    faithfulness: float
    completeness: float
    reasoning: str
    generated_answer: str
    ground_truth: str


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run"""
    name: str
    enable_rewrite: bool
    enable_web_search: bool
    use_unit_filter: bool
    description: str


# ============================================================================
# LLM-as-Judge Implementation
# ============================================================================

class LLMJudge:
    """LLM-based evaluation judge using external API"""
    
    JUDGE_PROMPT = """You are an expert evaluator for a Database Systems course Q&A system.
Evaluate the following system-generated answer against the ground truth.

## Question
{question}

## Ground Truth Answer
{ground_truth}

## System Generated Answer
{generated_answer}

## Retrieved Context (Source Documents)
{context}

## Evaluation Criteria (score 1-5 for each):

1. **Correctness**: Does the answer contain factually correct information that aligns with the ground truth?
   - 5: Completely correct, matches ground truth
   - 4: Mostly correct with minor omissions
   - 3: Partially correct, some errors
   - 2: Significant errors or misconceptions
   - 1: Largely incorrect

2. **Relevance**: Does the answer directly address the question asked?
   - 5: Directly and completely addresses the question
   - 4: Mostly relevant with minor tangents
   - 3: Somewhat relevant, missing key points
   - 2: Partially relevant, significant gaps
   - 1: Off-topic or irrelevant

3. **Faithfulness**: Is the answer grounded in the retrieved context (not hallucinated)?
   - 5: Fully grounded in retrieved documents
   - 4: Mostly grounded, minor extrapolation
   - 3: Partially grounded, some unsupported claims
   - 2: Significant hallucination
   - 1: Largely hallucinated

4. **Completeness**: Does the answer cover all important aspects mentioned in the ground truth?
   - 5: Covers all key points
   - 4: Covers most key points
   - 3: Covers some key points
   - 2: Missing many key points
   - 1: Very incomplete

Respond in JSON format only:
{{"correctness": <score>, "relevance": <score>, "faithfulness": <score>, "completeness": <score>, "reasoning": "<brief explanation>"}}
"""

    def __init__(self, api_type: str = "openai", model: str = None):
        """
        Initialize LLM Judge
        
        Args:
            api_type: "openai" or "anthropic"
            model: Model name (default: gpt-4o-mini for OpenAI, claude-3-haiku for Anthropic)
        """
        self.api_type = api_type
        
        if api_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            self.client = OpenAI()
            self.model = model or "gpt-4o-mini"
        elif api_type == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
            self.client = anthropic.Anthropic()
            self.model = model or "claude-3-haiku-20240307"
        else:
            raise ValueError(f"Unsupported API type: {api_type}")
    
    def evaluate(
        self,
        question: str,
        ground_truth: str,
        generated_answer: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Evaluate a single Q&A pair
        
        Returns:
            Dict with correctness, relevance, faithfulness, completeness, reasoning
        """
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            generated_answer=generated_answer,
            context=context[:2000] if context else "No context provided"
        )
        
        try:
            if self.api_type == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=500
                )
                result_text = response.choices[0].message.content
            else:  # anthropic
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                result_text = response.content[0].text
            
            # Parse JSON response
            # Handle potential markdown code blocks
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            result = json.loads(result_text.strip())
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse judge response: {e}")
            return {
                "correctness": 0,
                "relevance": 0,
                "faithfulness": 0,
                "completeness": 0,
                "reasoning": f"Parse error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return {
                "correctness": 0,
                "relevance": 0,
                "faithfulness": 0,
                "completeness": 0,
                "reasoning": f"API error: {str(e)}"
            }


# ============================================================================
# Retrieval Evaluator
# ============================================================================

class RetrievalEvaluator:
    """Evaluates retrieval quality"""
    
    def __init__(self, k: int = 4):
        self.k = k
    
    def evaluate_single(
        self,
        question_id: str,
        retrieved_docs: List[Any],
        expected_units: List[str],
        keywords: List[str]
    ) -> RetrievalResult:
        """
        Evaluate retrieval for a single query
        
        Args:
            retrieved_docs: List of Document objects with metadata
            expected_units: List of expected unit numbers (e.g., ["08", "09"])
            keywords: List of expected keywords in relevant documents
        """
        if not retrieved_docs:
            return RetrievalResult(
                question_id=question_id,
                precision_at_k=0.0,
                hit_rate=0.0,
                mrr=0.0,
                retrieved_units=[],
                expected_units=expected_units,
                keyword_matches=0,
                total_keywords=len(keywords)
            )
        
        # Extract unit numbers from retrieved docs
        retrieved_units = []
        relevance_scores = []
        first_relevant_rank = None
        keyword_match_count = 0
        
        for idx, doc in enumerate(retrieved_docs[:self.k]):
            metadata = getattr(doc, 'metadata', {}) or {}
            unit_no = metadata.get('unit_no') or metadata.get('unit_id', '')
            retrieved_units.append(str(unit_no))
            
            content = getattr(doc, 'page_content', '')
            content_lower = content.lower()
            
            # Check unit match
            unit_match = any(str(u).zfill(2) == str(unit_no).zfill(2) for u in expected_units)
            
            # Check keyword match
            keyword_match = any(kw.lower() in content_lower for kw in keywords)
            if keyword_match:
                keyword_match_count += 1
            
            is_relevant = unit_match or keyword_match
            relevance_scores.append(is_relevant)
            
            if is_relevant and first_relevant_rank is None:
                first_relevant_rank = idx + 1
        
        # Calculate metrics
        precision_at_k = sum(relevance_scores) / self.k if relevance_scores else 0.0
        hit_rate = 1.0 if any(relevance_scores) else 0.0
        mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        
        return RetrievalResult(
            question_id=question_id,
            precision_at_k=precision_at_k,
            hit_rate=hit_rate,
            mrr=mrr,
            retrieved_units=retrieved_units,
            expected_units=expected_units,
            keyword_matches=keyword_match_count,
            total_keywords=len(keywords)
        )
    
    def aggregate_results(self, results: List[RetrievalResult]) -> Dict[str, float]:
        """Aggregate multiple retrieval results"""
        if not results:
            return {}
        
        n = len(results)
        return {
            "avg_precision_at_k": sum(r.precision_at_k for r in results) / n,
            "avg_hit_rate": sum(r.hit_rate for r in results) / n,
            "avg_mrr": sum(r.mrr for r in results) / n,
            "total_questions": n
        }


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

class RAGEvaluator:
    """Main evaluation pipeline for TA Assistant RAG system"""
    
    def __init__(
        self,
        test_dataset_path: str,
        judge_api: str = "openai",
        judge_model: str = None,
        k: int = 4
    ):
        """
        Initialize evaluator
        
        Args:
            test_dataset_path: Path to test_dataset.json
            judge_api: API for LLM judge ("openai" or "anthropic")
            judge_model: Model name for judge
            k: Number of documents to retrieve
        """
        with open(test_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.test_cases = data['test_cases']
        self.metadata = data['metadata']
        self.judge = LLMJudge(api_type=judge_api, model=judge_model)
        self.retrieval_evaluator = RetrievalEvaluator(k=k)
        self.k = k
    
    def _import_ta_system(self):
        """Dynamically import TA Assistant system components"""
        try:
            # Add the TA system path if needed
            # sys.path.insert(0, '/path/to/ta_assistant')
            
            from run_TAassistant import (
                retrieval_qa_pipline,
                build_quiz_chains,
                QueryRewriter,
                get_embeddings,
                load_model
            )
            return {
                'retrieval_qa_pipline': retrieval_qa_pipline,
                'build_quiz_chains': build_quiz_chains,
                'QueryRewriter': QueryRewriter,
                'get_embeddings': get_embeddings,
                'load_model': load_model
            }
        except ImportError as e:
            logger.error(f"Failed to import TA system: {e}")
            return None
    
    def evaluate_retrieval_only(
        self,
        retriever,
        rewriter=None,
        use_korean: bool = False
    ) -> Tuple[List[RetrievalResult], Dict[str, float]]:
        """
        Evaluate retrieval component only
        
        Args:
            retriever: The retriever object
            rewriter: Optional QueryRewriter
            use_korean: Use Korean questions
        
        Returns:
            Tuple of (individual results, aggregated metrics)
        """
        results = []
        
        for case in self.test_cases:
            question = case['question_ko'] if use_korean else case['question']
            
            # Apply query rewriting if enabled
            search_query = question
            if rewriter:
                search_query = rewriter.rewrite(question)
            
            # Retrieve documents
            try:
                docs = retriever.get_relevant_documents(search_query)[:self.k]
            except Exception as e:
                logger.warning(f"Retrieval failed for {case['id']}: {e}")
                docs = []
            
            result = self.retrieval_evaluator.evaluate_single(
                question_id=case['id'],
                retrieved_docs=docs,
                expected_units=case.get('relevant_units', []),
                keywords=case.get('keywords', [])
            )
            results.append(result)
            logger.info(f"[{case['id']}] P@K={result.precision_at_k:.2f}, Hit={result.hit_rate:.0f}, MRR={result.mrr:.2f}")
        
        aggregated = self.retrieval_evaluator.aggregate_results(results)
        return results, aggregated
    
    def evaluate_generation(
        self,
        qa_chain,
        rewriter=None,
        use_korean: bool = False,
        max_workers: int = 4
    ) -> Tuple[List[GenerationResult], Dict[str, float]]:
        """
        Evaluate generation quality using LLM-as-Judge
        
        Args:
            qa_chain: The QA chain object
            rewriter: Optional QueryRewriter
            use_korean: Use Korean questions
            max_workers: Number of parallel workers for API calls
        
        Returns:
            Tuple of (individual results, aggregated metrics)
        """
        results = []
        
        for case in self.test_cases:
            question = case['question_ko'] if use_korean else case['question']
            
            # Apply query rewriting if enabled
            search_query = question
            if rewriter:
                search_query = rewriter.rewrite(question)
            
            # Generate answer
            try:
                response = qa_chain.invoke(search_query)
                generated_answer = response.get('result', '')
                source_docs = response.get('source_documents', [])
                context = "\n\n".join([
                    f"[Source: {getattr(d, 'metadata', {}).get('source', 'unknown')}]\n{getattr(d, 'page_content', '')}"
                    for d in source_docs
                ])
            except Exception as e:
                logger.warning(f"Generation failed for {case['id']}: {e}")
                generated_answer = ""
                context = ""
            
            # Evaluate with LLM judge
            judge_result = self.judge.evaluate(
                question=question,
                ground_truth=case['ground_truth'],
                generated_answer=generated_answer,
                context=context
            )
            
            result = GenerationResult(
                question_id=case['id'],
                correctness=judge_result.get('correctness', 0),
                relevance=judge_result.get('relevance', 0),
                faithfulness=judge_result.get('faithfulness', 0),
                completeness=judge_result.get('completeness', 0),
                reasoning=judge_result.get('reasoning', ''),
                generated_answer=generated_answer,
                ground_truth=case['ground_truth']
            )
            results.append(result)
            logger.info(
                f"[{case['id']}] Correct={result.correctness}, "
                f"Rel={result.relevance}, Faith={result.faithfulness}"
            )
            
            # Rate limiting
            time.sleep(0.5)
        
        # Aggregate results
        n = len(results)
        aggregated = {
            "avg_correctness": sum(r.correctness for r in results) / n,
            "avg_relevance": sum(r.relevance for r in results) / n,
            "avg_faithfulness": sum(r.faithfulness for r in results) / n,
            "avg_completeness": sum(r.completeness for r in results) / n,
            "total_questions": n
        }
        
        return results, aggregated
    
    def run_full_evaluation(
        self,
        config: EvaluationConfig,
        device_type: str = "cuda",
        use_korean: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete evaluation with given configuration
        
        Args:
            config: Evaluation configuration
            device_type: Device type for model loading
            use_korean: Whether to use Korean questions
        
        Returns:
            Complete evaluation results
        """
        logger.info(f"Starting evaluation: {config.name}")
        logger.info(f"Config: rewrite={config.enable_rewrite}, web={config.enable_web_search}, filter={config.use_unit_filter}")
        
        start_time = time.time()
        
        # Import TA system
        ta_system = self._import_ta_system()
        if ta_system is None:
            return {"error": "Failed to import TA system"}
        
        # Build QA chain
        qa_chain, rewriter = ta_system['retrieval_qa_pipline'](
            device_type=device_type,
            use_history=False,
            mode="answer",
            enable_web_search=config.enable_web_search,
            enable_rewrite=config.enable_rewrite
        )
        
        # Get retriever from chain
        retriever = qa_chain.retriever
        
        # Evaluate retrieval
        retrieval_results, retrieval_metrics = self.evaluate_retrieval_only(
            retriever=retriever,
            rewriter=rewriter if config.enable_rewrite else None,
            use_korean=use_korean
        )
        
        # Evaluate generation
        generation_results, generation_metrics = self.evaluate_generation(
            qa_chain=qa_chain,
            rewriter=rewriter if config.enable_rewrite else None,
            use_korean=use_korean
        )
        
        elapsed_time = time.time() - start_time
        
        return {
            "config": asdict(config) if hasattr(config, '__dataclass_fields__') else config.__dict__,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed_time,
            "retrieval": {
                "metrics": retrieval_metrics,
                "details": [asdict(r) for r in retrieval_results]
            },
            "generation": {
                "metrics": generation_metrics,
                "details": [asdict(r) for r in generation_results]
            }
        }


# ============================================================================
# Ablation Study Runner
# ============================================================================

def run_ablation_study(
    evaluator: RAGEvaluator,
    output_path: str,
    device_type: str = "cuda"
) -> Dict[str, Any]:
    """
    Run ablation study across different configurations
    
    Args:
        evaluator: RAGEvaluator instance
        output_path: Path to save results
        device_type: Device type for model loading
    
    Returns:
        Complete ablation study results
    """
    configurations = [
        EvaluationConfig(
            name="Baseline",
            enable_rewrite=False,
            enable_web_search=False,
            use_unit_filter=False,
            description="Basic RAG without any enhancements"
        ),
        EvaluationConfig(
            name="+Rewrite",
            enable_rewrite=True,
            enable_web_search=False,
            use_unit_filter=False,
            description="Baseline + Query Rewriting"
        ),
        EvaluationConfig(
            name="+Web",
            enable_rewrite=False,
            enable_web_search=True,
            use_unit_filter=False,
            description="Baseline + Web Search Fallback"
        ),
        EvaluationConfig(
            name="Full",
            enable_rewrite=True,
            enable_web_search=True,
            use_unit_filter=True,
            description="All features enabled"
        ),
    ]
    
    all_results = {
        "study_type": "ablation",
        "timestamp": datetime.now().isoformat(),
        "configurations": []
    }
    
    for config in configurations:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running configuration: {config.name}")
        logger.info(f"{'='*60}\n")
        
        result = evaluator.run_full_evaluation(
            config=config,
            device_type=device_type,
            use_korean=True
        )
        all_results["configurations"].append(result)
        
        # Save intermediate results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved intermediate results to {output_path}")
    
    # Generate summary table
    summary = generate_summary_table(all_results)
    all_results["summary"] = summary
    
    # Save final results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    return all_results


def generate_summary_table(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary comparison table from ablation results"""
    summary = {
        "retrieval_comparison": [],
        "generation_comparison": []
    }
    
    for config_result in results.get("configurations", []):
        config = config_result.get("config", {})
        ret_metrics = config_result.get("retrieval", {}).get("metrics", {})
        gen_metrics = config_result.get("generation", {}).get("metrics", {})
        
        summary["retrieval_comparison"].append({
            "name": config.get("name", "Unknown"),
            "precision_at_k": ret_metrics.get("avg_precision_at_k", 0),
            "hit_rate": ret_metrics.get("avg_hit_rate", 0),
            "mrr": ret_metrics.get("avg_mrr", 0)
        })
        
        summary["generation_comparison"].append({
            "name": config.get("name", "Unknown"),
            "correctness": gen_metrics.get("avg_correctness", 0),
            "relevance": gen_metrics.get("avg_relevance", 0),
            "faithfulness": gen_metrics.get("avg_faithfulness", 0),
            "completeness": gen_metrics.get("avg_completeness", 0)
        })
    
    return summary


# ============================================================================
# Standalone Mode (without TA system - for testing evaluation logic)
# ============================================================================

class MockRetriever:
    """Mock retriever for testing evaluation logic"""
    
    def __init__(self, docs: List[Dict] = None):
        self.docs = docs or []
    
    def get_relevant_documents(self, query: str):
        """Return mock documents"""
        class MockDoc:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata
        
        return [
            MockDoc(
                content=f"This is mock content about {query}. B+ tree is a balanced tree.",
                metadata={"source": "mock.pdf", "unit_no": "08"}
            )
            for _ in range(4)
        ]


class MockQAChain:
    """Mock QA chain for testing evaluation logic"""
    
    def __init__(self, retriever=None):
        self.retriever = retriever or MockRetriever()
    
    def invoke(self, query: str):
        """Return mock response"""
        docs = self.retriever.get_relevant_documents(query)
        return {
            "result": f"This is a mock answer about {query}. The answer includes information about database concepts.",
            "source_documents": docs
        }


def run_standalone_test(
    test_dataset_path: str,
    output_path: str,
    judge_api: str = "openai"
):
    """
    Run evaluation in standalone mode (without actual TA system)
    Useful for testing evaluation logic
    """
    logger.info("Running in standalone test mode with mock components")
    
    evaluator = RAGEvaluator(
        test_dataset_path=test_dataset_path,
        judge_api=judge_api
    )
    
    # Use mock components
    mock_retriever = MockRetriever()
    mock_qa = MockQAChain(retriever=mock_retriever)
    
    # Evaluate retrieval
    retrieval_results, retrieval_metrics = evaluator.evaluate_retrieval_only(
        retriever=mock_retriever,
        rewriter=None,
        use_korean=False
    )
    
    # Evaluate generation (only first 5 for cost saving in test mode)
    evaluator.test_cases = evaluator.test_cases[:5]
    generation_results, generation_metrics = evaluator.evaluate_generation(
        qa_chain=mock_qa,
        rewriter=None,
        use_korean=False
    )
    
    results = {
        "mode": "standalone_test",
        "timestamp": datetime.now().isoformat(),
        "retrieval": {
            "metrics": retrieval_metrics,
            "sample_results": [asdict(r) for r in retrieval_results[:5]]
        },
        "generation": {
            "metrics": generation_metrics,
            "sample_results": [asdict(r) for r in generation_results]
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Standalone test results saved to {output_path}")
    return results


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RAG System Evaluation for TA Assistant")
    parser.add_argument(
        "--dataset",
        type=str,
        default="test_dataset_textbook.json",
        help="Path to test dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results (default: <dataset_stem>_results.json)"
    )
    parser.add_argument(
        "--judge-api",
        type=str,
        choices=["openai", "anthropic"],
        default="openai",
        help="API to use for LLM-as-Judge"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model name for judge (default: gpt-4o-mini or claude-3-haiku)"
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run full ablation study"
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Run in standalone test mode (no TA system required)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device type (cuda, cpu, etc.)"
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["baseline", "rewrite", "web", "full"],
        default="full",
        help="Single configuration to evaluate"
    )
    
    args = parser.parse_args()
    
    dataset_stem = Path(args.dataset).stem
    if not args.output:
        args.output = f"{dataset_stem}_results.json"

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.standalone:
        # Run standalone test
        run_standalone_test(
            test_dataset_path=args.dataset,
            output_path=args.output,
            judge_api=args.judge_api
        )
    elif args.ablation:
        # Run ablation study
        evaluator = RAGEvaluator(
            test_dataset_path=args.dataset,
            judge_api=args.judge_api,
            judge_model=args.judge_model
        )
        run_ablation_study(
            evaluator=evaluator,
            output_path=args.output,
            device_type=args.device
        )
    else:
        # Run single configuration
        config_map = {
            "baseline": EvaluationConfig(
                name="Baseline",
                enable_rewrite=False,
                enable_web_search=False,
                use_unit_filter=False,
                description="Basic RAG"
            ),
            "rewrite": EvaluationConfig(
                name="+Rewrite",
                enable_rewrite=True,
                enable_web_search=False,
                use_unit_filter=False,
                description="With Query Rewriting"
            ),
            "web": EvaluationConfig(
                name="+Web",
                enable_rewrite=False,
                enable_web_search=True,
                use_unit_filter=False,
                description="With Web Search"
            ),
            "full": EvaluationConfig(
                name="Full",
                enable_rewrite=True,
                enable_web_search=True,
                use_unit_filter=True,
                description="All features"
            ),
        }
        
        evaluator = RAGEvaluator(
            test_dataset_path=args.dataset,
            judge_api=args.judge_api,
            judge_model=args.judge_model
        )
        
        result = evaluator.run_full_evaluation(
            config=config_map[args.config],
            device_type=args.device,
            use_korean=True
        )
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
