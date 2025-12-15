"""
Evaluation Results Analysis and Visualization

This script analyzes evaluation results and generates:
1. Summary tables
2. Comparison charts
3. Error analysis reports

Usage:
    python analyze_results.py --input results/ablation_study.json --output-dir analysis/
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def load_results(filepath: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_markdown_report(results: Dict[str, Any], output_path: str):
    """Generate a comprehensive markdown report"""
    
    report_lines = [
        "# RAG System Evaluation Report",
        "",
        f"**Generated:** {results.get('timestamp', 'N/A')}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
    ]
    
    # Check if this is an ablation study
    if "configurations" in results:
        configs = results["configurations"]
        
        # Find best configuration
        best_retrieval = max(configs, key=lambda x: x.get("retrieval", {}).get("metrics", {}).get("avg_precision_at_k", 0))
        best_generation = max(configs, key=lambda x: x.get("generation", {}).get("metrics", {}).get("avg_correctness", 0))
        
        report_lines.extend([
            f"- **Best Retrieval Performance:** {best_retrieval.get('config', {}).get('name', 'N/A')}",
            f"- **Best Generation Performance:** {best_generation.get('config', {}).get('name', 'N/A')}",
            "",
        ])
        
        # Retrieval comparison table
        report_lines.extend([
            "## 2. Retrieval Performance Comparison",
            "",
            "| Configuration | Precision@K | Hit Rate | MRR |",
            "|--------------|-------------|----------|-----|",
        ])
        
        for config in configs:
            name = config.get("config", {}).get("name", "Unknown")
            metrics = config.get("retrieval", {}).get("metrics", {})
            report_lines.append(
                f"| {name} | {metrics.get('avg_precision_at_k', 0):.3f} | "
                f"{metrics.get('avg_hit_rate', 0):.3f} | {metrics.get('avg_mrr', 0):.3f} |"
            )
        
        report_lines.extend([
            "",
            "## 3. Generation Performance Comparison",
            "",
            "| Configuration | Correctness | Relevance | Faithfulness | Completeness |",
            "|--------------|-------------|-----------|--------------|--------------|",
        ])
        
        for config in configs:
            name = config.get("config", {}).get("name", "Unknown")
            metrics = config.get("generation", {}).get("metrics", {})
            report_lines.append(
                f"| {name} | {metrics.get('avg_correctness', 0):.2f} | "
                f"{metrics.get('avg_relevance', 0):.2f} | "
                f"{metrics.get('avg_faithfulness', 0):.2f} | "
                f"{metrics.get('avg_completeness', 0):.2f} |"
            )
        
        # Analysis section
        report_lines.extend([
            "",
            "## 4. Analysis",
            "",
            "### 4.1 Feature Contribution Analysis",
            "",
        ])
        
        # Calculate feature contributions if we have baseline
        baseline = next((c for c in configs if "baseline" in c.get("config", {}).get("name", "").lower()), None)
        
        if baseline:
            baseline_p = baseline.get("retrieval", {}).get("metrics", {}).get("avg_precision_at_k", 0)
            baseline_c = baseline.get("generation", {}).get("metrics", {}).get("avg_correctness", 0)
            
            report_lines.append("**Retrieval Improvement over Baseline:**")
            report_lines.append("")
            
            for config in configs:
                name = config.get("config", {}).get("name", "Unknown")
                if "baseline" in name.lower():
                    continue
                p = config.get("retrieval", {}).get("metrics", {}).get("avg_precision_at_k", 0)
                improvement = ((p - baseline_p) / baseline_p * 100) if baseline_p > 0 else 0
                report_lines.append(f"- **{name}**: {improvement:+.1f}% in Precision@K")
            
            report_lines.extend([
                "",
                "**Generation Improvement over Baseline:**",
                ""
            ])
            
            for config in configs:
                name = config.get("config", {}).get("name", "Unknown")
                if "baseline" in name.lower():
                    continue
                c = config.get("generation", {}).get("metrics", {}).get("avg_correctness", 0)
                improvement = ((c - baseline_c) / baseline_c * 100) if baseline_c > 0 else 0
                report_lines.append(f"- **{name}**: {improvement:+.1f}% in Correctness")
        
        # Error analysis
        report_lines.extend([
            "",
            "### 4.2 Error Analysis",
            "",
        ])
        
        # Find questions with lowest scores
        full_config = next((c for c in configs if "full" in c.get("config", {}).get("name", "").lower()), configs[-1])
        gen_details = full_config.get("generation", {}).get("details", [])
        
        if gen_details:
            low_scoring = sorted(gen_details, key=lambda x: x.get("correctness", 0))[:5]
            report_lines.extend([
                "**Lowest Scoring Questions (Full Configuration):**",
                "",
                "| Question ID | Correctness | Reasoning |",
                "|-------------|-------------|-----------|",
            ])
            for item in low_scoring:
                reasoning = item.get("reasoning", "")[:50] + "..." if len(item.get("reasoning", "")) > 50 else item.get("reasoning", "")
                report_lines.append(
                    f"| {item.get('question_id', 'N/A')} | "
                    f"{item.get('correctness', 0):.1f} | {reasoning} |"
                )
    
    else:
        # Single configuration result
        ret_metrics = results.get("retrieval", {}).get("metrics", {})
        gen_metrics = results.get("generation", {}).get("metrics", {})
        
        report_lines.extend([
            "### Retrieval Metrics",
            "",
            f"- Precision@K: {ret_metrics.get('avg_precision_at_k', 0):.3f}",
            f"- Hit Rate: {ret_metrics.get('avg_hit_rate', 0):.3f}",
            f"- MRR: {ret_metrics.get('avg_mrr', 0):.3f}",
            "",
            "### Generation Metrics",
            "",
            f"- Correctness: {gen_metrics.get('avg_correctness', 0):.2f}/5",
            f"- Relevance: {gen_metrics.get('avg_relevance', 0):.2f}/5",
            f"- Faithfulness: {gen_metrics.get('avg_faithfulness', 0):.2f}/5",
            f"- Completeness: {gen_metrics.get('avg_completeness', 0):.2f}/5",
        ])
    
    # Recommendations
    report_lines.extend([
        "",
        "## 5. Recommendations",
        "",
        "Based on the evaluation results:",
        "",
        "1. **Query Rewriting**: ",
        "   - Evaluate impact on Korean vs English queries",
        "   - Consider domain-specific rewriting rules",
        "",
        "2. **Web Search Fallback**:",
        "   - Monitor hallucination rates when web search is enabled",
        "   - Consider relevance filtering for web results",
        "",
        "3. **Future Improvements**:",
        "   - Implement re-ranking for retrieved documents",
        "   - Add query decomposition for complex questions",
        "   - Consider fine-tuning embedding model on course materials",
        "",
        "---",
        "",
        "*Report generated by RAG Evaluation System*"
    ])
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Report saved to {output_path}")


def calculate_std_from_details(config: Dict, metric_type: str, metric_name: str) -> float:
    """Calculate standard deviation from detailed results"""
    details = config.get(metric_type, {}).get("details", [])
    if not details:
        return 0.0
    
    # Map metric names to detail field names
    field_map = {
        "Precision@K": "precision_at_k",
        "Hit Rate": "hit_rate", 
        "MRR": "mrr",
        "Correctness": "correctness",
        "Relevance": "relevance",
        "Faithfulness": "faithfulness",
        "Completeness": "completeness"
    }
    
    field = field_map.get(metric_name, metric_name.lower())
    values = [d.get(field, 0) for d in details]
    
    if not values:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def generate_charts(results: Dict[str, Any], output_dir: str):
    """Generate visualization charts"""
    
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available. Skipping chart generation.")
        return
    
    if "configurations" not in results:
        print("No ablation study data found. Skipping charts.")
        return
    
    configs = results["configurations"]
    
    # Prepare data
    names = [c.get("config", {}).get("name", "Unknown") for c in configs]
    
    retrieval_data = {
        "Precision@K": [c.get("retrieval", {}).get("metrics", {}).get("avg_precision_at_k", 0) for c in configs],
        "Hit Rate": [c.get("retrieval", {}).get("metrics", {}).get("avg_hit_rate", 0) for c in configs],
        "MRR": [c.get("retrieval", {}).get("metrics", {}).get("avg_mrr", 0) for c in configs],
    }
    
    generation_data = {
        "Correctness": [c.get("generation", {}).get("metrics", {}).get("avg_correctness", 0) for c in configs],
        "Relevance": [c.get("generation", {}).get("metrics", {}).get("avg_relevance", 0) for c in configs],
        "Faithfulness": [c.get("generation", {}).get("metrics", {}).get("avg_faithfulness", 0) for c in configs],
        "Completeness": [c.get("generation", {}).get("metrics", {}).get("avg_completeness", 0) for c in configs],
    }
    
    # Calculate standard deviations for error bars
    retrieval_std = {
        metric: [calculate_std_from_details(c, "retrieval", metric) for c in configs]
        for metric in retrieval_data.keys()
    }
    
    generation_std = {
        metric: [calculate_std_from_details(c, "generation", metric) for c in configs]
        for metric in generation_data.keys()
    }
    
    # =========================================================================
    # Chart 1: Combined 3-subplot chart (as requested)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = range(len(names))
    
    # Subplot 1: Retrieval Performance with error bars
    width = 0.25
    colors_ret = ['#2ecc71', '#3498db', '#9b59b6']
    for i, (metric, values) in enumerate(retrieval_data.items()):
        bars = axes[0].bar(
            [xi + i * width for xi in x], 
            values, 
            width, 
            label=metric,
            color=colors_ret[i],
            yerr=retrieval_std[metric],
            capsize=3,
            error_kw={'elinewidth': 1, 'capthick': 1}
        )
    
    axes[0].set_xlabel('Configuration')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Retrieval Performance')
    axes[0].set_xticks([xi + width for xi in x])
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Subplot 2: Generation Performance with error bars
    width = 0.18
    colors_gen = ['#e74c3c', '#f39c12', '#1abc9c', '#8e44ad']
    for i, (metric, values) in enumerate(generation_data.items()):
        bars = axes[1].bar(
            [xi + i * width for xi in x], 
            values, 
            width, 
            label=metric,
            color=colors_gen[i],
            yerr=generation_std[metric],
            capsize=2,
            error_kw={'elinewidth': 1, 'capthick': 1}
        )
    
    axes[1].set_xlabel('Configuration')
    axes[1].set_ylabel('Score (1-5)')
    axes[1].set_title('Generation Performance')
    axes[1].set_xticks([xi + 1.5 * width for xi in x])
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].legend(loc='upper left', fontsize=8)
    axes[1].set_ylim(0, 5.5)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Subplot 3: Ablation Contribution (improvement over baseline)
    if len(configs) >= 2:
        baseline_ret = retrieval_data['Precision@K'][0]
        baseline_gen = generation_data['Correctness'][0]
        
        improvements_ret = []
        improvements_gen = []
        config_names = []
        
        for i, name in enumerate(names):
            if i == 0:  # Skip baseline
                continue
            config_names.append(name)
            
            # Calculate relative improvement
            ret_imp = ((retrieval_data['Precision@K'][i] - baseline_ret) / baseline_ret * 100) if baseline_ret > 0 else 0
            gen_imp = ((generation_data['Correctness'][i] - baseline_gen) / baseline_gen * 100) if baseline_gen > 0 else 0
            
            improvements_ret.append(ret_imp)
            improvements_gen.append(gen_imp)
        
        x_contrib = range(len(config_names))
        width = 0.35
        
        bars1 = axes[2].bar([xi - width/2 for xi in x_contrib], improvements_ret, width, 
                           label='Retrieval (Precision@K)', color='#3498db')
        bars2 = axes[2].bar([xi + width/2 for xi in x_contrib], improvements_gen, width,
                           label='Generation (Correctness)', color='#e74c3c')
        
        axes[2].set_xlabel('Configuration')
        axes[2].set_ylabel('Improvement over Baseline (%)')
        axes[2].set_title('Ablation: Feature Contribution')
        axes[2].set_xticks(x_contrib)
        axes[2].set_xticklabels(config_names, rotation=45, ha='right')
        axes[2].legend(loc='upper left', fontsize=8)
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[2].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[2].annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            height = bar.get_height()
            axes[2].annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_results.png'), dpi=300)
    plt.close()
    
    # =========================================================================
    # Chart 2: Individual charts (for flexibility)
    # =========================================================================
    
    # Retrieval only
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.25
    for i, (metric, values) in enumerate(retrieval_data.items()):
        ax.bar([xi + i * width for xi in x], values, width, label=metric,
               color=colors_ret[i], yerr=retrieval_std[metric], capsize=3)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Score')
    ax.set_title('Retrieval Performance Comparison')
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'retrieval_comparison.png'), dpi=150)
    plt.close()
    
    # Generation only
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.18
    for i, (metric, values) in enumerate(generation_data.items()):
        ax.bar([xi + i * width for xi in x], values, width, label=metric,
               color=colors_gen[i], yerr=generation_std[metric], capsize=2)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Score (1-5)')
    ax.set_title('Generation Performance Comparison')
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 5.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generation_comparison.png'), dpi=150)
    plt.close()
    
    # =========================================================================
    # Chart 3: Radar Chart (Baseline vs Full)
    # =========================================================================
    if len(configs) >= 2:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        categories = ['Precision@K', 'Hit Rate', 'MRR', 'Correctness\n(norm)', 'Relevance\n(norm)', 'Faithfulness\n(norm)']
        
        baseline_idx = 0
        full_idx = -1
        
        baseline_values = [
            retrieval_data['Precision@K'][baseline_idx],
            retrieval_data['Hit Rate'][baseline_idx],
            retrieval_data['MRR'][baseline_idx],
            generation_data['Correctness'][baseline_idx] / 5,
            generation_data['Relevance'][baseline_idx] / 5,
            generation_data['Faithfulness'][baseline_idx] / 5,
        ]
        
        full_values = [
            retrieval_data['Precision@K'][full_idx],
            retrieval_data['Hit Rate'][full_idx],
            retrieval_data['MRR'][full_idx],
            generation_data['Correctness'][full_idx] / 5,
            generation_data['Relevance'][full_idx] / 5,
            generation_data['Faithfulness'][full_idx] / 5,
        ]
        
        # Close the polygon
        baseline_values += baseline_values[:1]
        full_values += full_values[:1]
        
        angles = [n / float(len(categories)) * 2 * 3.14159 for n in range(len(categories))]
        angles += angles[:1]
        
        ax.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline', color='#3498db')
        ax.fill(angles, baseline_values, alpha=0.25, color='#3498db')
        ax.plot(angles, full_values, 'o-', linewidth=2, label='Full', color='#e74c3c')
        ax.fill(angles, full_values, alpha=0.25, color='#e74c3c')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title('Baseline vs Full System\n(Multi-dimensional Comparison)', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=150)
        plt.close()
    
    # =========================================================================
    # Chart 4: Heatmap for detailed comparison
    # =========================================================================
    all_metrics = []
    for config in configs:
        name = config.get("config", {}).get("name", "Unknown")
        ret = config.get("retrieval", {}).get("metrics", {})
        gen = config.get("generation", {}).get("metrics", {})
        all_metrics.append({
            "Config": name,
            "P@K": ret.get("avg_precision_at_k", 0),
            "Hit": ret.get("avg_hit_rate", 0),
            "MRR": ret.get("avg_mrr", 0),
            "Corr": gen.get("avg_correctness", 0) / 5,  # Normalize to 0-1
            "Rel": gen.get("avg_relevance", 0) / 5,
            "Faith": gen.get("avg_faithfulness", 0) / 5,
            "Comp": gen.get("avg_completeness", 0) / 5,
        })
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    metric_names = ["P@K", "Hit", "MRR", "Corr", "Rel", "Faith", "Comp"]
    data_matrix = [[m[k] for k in metric_names] for m in all_metrics]
    config_names = [m["Config"] for m in all_metrics]
    
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names)
    ax.set_yticks(range(len(config_names)))
    ax.set_yticklabels(config_names)
    
    # Add value annotations
    for i in range(len(config_names)):
        for j in range(len(metric_names)):
            val = data_matrix[i][j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Score (normalized 0-1)')
    ax.set_title('Performance Heatmap (All Metrics Normalized)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_heatmap.png'), dpi=150)
    plt.close()
    
    print(f"Charts saved to {output_dir}")
    print(f"  - evaluation_results.png (combined 3-subplot)")
    print(f"  - retrieval_comparison.png")
    print(f"  - generation_comparison.png")
    print(f"  - radar_comparison.png")
    print(f"  - performance_heatmap.png")


def generate_csv_export(results: Dict[str, Any], output_dir: str):
    """Export results to CSV format"""
    
    if not PANDAS_AVAILABLE:
        print("pandas not available. Skipping CSV export.")
        # Fallback: manual CSV generation
        if "configurations" in results:
            # Retrieval CSV
            with open(os.path.join(output_dir, 'retrieval_results.csv'), 'w', encoding='utf-8') as f:
                f.write("Configuration,Precision@K,Hit Rate,MRR\n")
                for config in results["configurations"]:
                    name = config.get("config", {}).get("name", "Unknown")
                    metrics = config.get("retrieval", {}).get("metrics", {})
                    f.write(f"{name},{metrics.get('avg_precision_at_k', 0):.4f},"
                           f"{metrics.get('avg_hit_rate', 0):.4f},{metrics.get('avg_mrr', 0):.4f}\n")
            
            # Generation CSV
            with open(os.path.join(output_dir, 'generation_results.csv'), 'w', encoding='utf-8') as f:
                f.write("Configuration,Correctness,Relevance,Faithfulness,Completeness\n")
                for config in results["configurations"]:
                    name = config.get("config", {}).get("name", "Unknown")
                    metrics = config.get("generation", {}).get("metrics", {})
                    f.write(f"{name},{metrics.get('avg_correctness', 0):.4f},"
                           f"{metrics.get('avg_relevance', 0):.4f},"
                           f"{metrics.get('avg_faithfulness', 0):.4f},"
                           f"{metrics.get('avg_completeness', 0):.4f}\n")
        return
    
    if "configurations" in results:
        # Retrieval results
        ret_data = []
        for config in results["configurations"]:
            name = config.get("config", {}).get("name", "Unknown")
            metrics = config.get("retrieval", {}).get("metrics", {})
            ret_data.append({
                "Configuration": name,
                "Precision@K": metrics.get("avg_precision_at_k", 0),
                "Hit Rate": metrics.get("avg_hit_rate", 0),
                "MRR": metrics.get("avg_mrr", 0)
            })
        
        pd.DataFrame(ret_data).to_csv(
            os.path.join(output_dir, 'retrieval_results.csv'),
            index=False
        )
        
        # Generation results
        gen_data = []
        for config in results["configurations"]:
            name = config.get("config", {}).get("name", "Unknown")
            metrics = config.get("generation", {}).get("metrics", {})
            gen_data.append({
                "Configuration": name,
                "Correctness": metrics.get("avg_correctness", 0),
                "Relevance": metrics.get("avg_relevance", 0),
                "Faithfulness": metrics.get("avg_faithfulness", 0),
                "Completeness": metrics.get("avg_completeness", 0)
            })
        
        pd.DataFrame(gen_data).to_csv(
            os.path.join(output_dir, 'generation_results.csv'),
            index=False
        )
        
        # Detailed results per question
        for config in results["configurations"]:
            name = config.get("config", {}).get("name", "Unknown").replace("+", "plus_")
            
            # Retrieval details
            ret_details = config.get("retrieval", {}).get("details", [])
            if ret_details:
                pd.DataFrame(ret_details).to_csv(
                    os.path.join(output_dir, f'retrieval_details_{name}.csv'),
                    index=False
                )
            
            # Generation details
            gen_details = config.get("generation", {}).get("details", [])
            if gen_details:
                pd.DataFrame(gen_details).to_csv(
                    os.path.join(output_dir, f'generation_details_{name}.csv'),
                    index=False
                )
    
    print(f"CSV files saved to {output_dir}")


def analyze_by_category(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze results broken down by question category"""
    
    if "configurations" not in results:
        return {}
    
    # Get the full configuration for detailed analysis
    full_config = next(
        (c for c in results["configurations"] if "full" in c.get("config", {}).get("name", "").lower()),
        results["configurations"][-1]
    )
    
    gen_details = full_config.get("generation", {}).get("details", [])
    
    # Load test dataset to get categories
    # This assumes test_dataset.json is in the same directory
    try:
        with open("test_dataset.json", 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        id_to_category = {tc["id"]: tc["category"] for tc in dataset["test_cases"]}
    except:
        # If dataset not available, return empty
        return {}
    
    # Group by category
    category_scores = defaultdict(lambda: {"correctness": [], "relevance": [], "faithfulness": []})
    
    for detail in gen_details:
        qid = detail.get("question_id")
        category = id_to_category.get(qid, "unknown")
        category_scores[category]["correctness"].append(detail.get("correctness", 0))
        category_scores[category]["relevance"].append(detail.get("relevance", 0))
        category_scores[category]["faithfulness"].append(detail.get("faithfulness", 0))
    
    # Calculate averages
    category_analysis = {}
    for category, scores in category_scores.items():
        n = len(scores["correctness"])
        category_analysis[category] = {
            "count": n,
            "avg_correctness": sum(scores["correctness"]) / n if n > 0 else 0,
            "avg_relevance": sum(scores["relevance"]) / n if n > 0 else 0,
            "avg_faithfulness": sum(scores["faithfulness"]) / n if n > 0 else 0,
        }
    
    return category_analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze RAG evaluation results")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to evaluation results JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis",
        help="Directory to save analysis outputs"
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.input)
    
    # Generate markdown report (name follows input stem)
    result_stem = Path(args.input).stem
    generate_markdown_report(
        results,
        os.path.join(args.output_dir, f"{result_stem}_report.md")
    )
    
    # Generate charts
    if not args.no_charts:
        generate_charts(results, args.output_dir)
    
    # Export to CSV
    generate_csv_export(results, args.output_dir)
    
    # Category analysis
    category_analysis = analyze_by_category(results)
    if category_analysis:
        with open(os.path.join(args.output_dir, "category_analysis.json"), 'w', encoding='utf-8') as f:
            json.dump(category_analysis, f, ensure_ascii=False, indent=2)
        print(f"Category analysis saved to {args.output_dir}/category_analysis.json")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
