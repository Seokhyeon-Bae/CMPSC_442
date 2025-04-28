#!/usr/bin/env python3
# ROUGE Evaluation Script for CMPSC 442 Homework 6

import json
import argparse
import numpy as np
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import nltk
import os

# Download NLTK resources (if needed)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')

def load_results(file_path):
    """Load evaluation results from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_rouge_scores(results):
    """Calculate ROUGE scores for all examples."""
    # Create ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Initialize score lists
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    # Track token lengths
    generated_lengths = []
    reference_lengths = []
    
    # Process each example
    scores_by_example = []
    
    for i, example in enumerate(results):
        generated = example['generated']
        reference = example['reference']
        
        # Skip empty generations/references
        if not generated or not reference:
            continue
        
        # Calculate scores
        scores = scorer.score(reference, generated)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
        
        # Calculate token lengths
        generated_tokens = word_tokenize(generated)
        reference_tokens = word_tokenize(reference)
        generated_lengths.append(len(generated_tokens))
        reference_lengths.append(len(reference_tokens))
        
        scores_by_example.append({
            'id': i,
            'generated': generated,
            'reference': reference,
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure,
            'gen_tokens': len(generated_tokens),
            'ref_tokens': len(reference_tokens),
        })
    
    # Calculate averages
    avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = np.mean(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = np.mean(rougeL_scores) if rougeL_scores else 0
    
    avg_gen_length = np.mean(generated_lengths) if generated_lengths else 0
    avg_ref_length = np.mean(reference_lengths) if reference_lengths else 0
    
    # Sort by ROUGE-L for best/worst examples
    scores_by_example.sort(key=lambda x: x['rougeL'])
    
    return {
        'averages': {
            'rouge1': avg_rouge1,
            'rouge2': avg_rouge2,
            'rougeL': avg_rougeL,
            'gen_length': avg_gen_length,
            'ref_length': avg_ref_length
        },
        'scores': {
            'rouge1': rouge1_scores,
            'rouge2': rouge2_scores,
            'rougeL': rougeL_scores,
        },
        'all_scores': scores_by_example,
        'lengths': {
            'generated': generated_lengths,
            'reference': reference_lengths
        }
    }

def print_report(eval_results):
    """Print a comprehensive report of the evaluation results."""
    averages = eval_results['averages']
    all_scores = eval_results['all_scores']
    
    print("\n=== ROUGE Score Summary ===")
    print(f"Average ROUGE-1: {averages['rouge1']:.4f}")
    print(f"Average ROUGE-2: {averages['rouge2']:.4f}")
    print(f"Average ROUGE-L: {averages['rougeL']:.4f}")
    
    print(f"\nAverage Generated Length: {averages['gen_length']:.1f} tokens")
    print(f"Average Reference Length: {averages['ref_length']:.1f} tokens")
    
    # Score distribution
    r1 = eval_results['scores']['rouge1']
    r2 = eval_results['scores']['rouge2']
    rl = eval_results['scores']['rougeL']
    
    print("\n=== ROUGE Score Distribution ===")
    for name, scores in [("ROUGE-1", r1), ("ROUGE-2", r2), ("ROUGE-L", rl)]:
        print(f"\n{name}:")
        print(f"  Min: {min(scores) if scores else 0:.4f}")
        print(f"  Max: {max(scores) if scores else 0:.4f}")
        print(f"  Median: {np.median(scores) if scores else 0:.4f}")
        print(f"  Std Dev: {np.std(scores) if scores else 0:.4f}")
    
    # Print worst examples
    print("\n=== Worst Performing Examples ===")
    for i, example in enumerate(all_scores[:3]):
        print(f"\nExample {example['id']}:")
        print(f"ROUGE-L Score: {example['rougeL']:.4f}")
        print(f"Generated ({example['gen_tokens']} tokens): {example['generated']}")
        print(f"Reference ({example['ref_tokens']} tokens): {example['reference']}")
    
    # Print best examples
    print("\n=== Best Performing Examples ===")
    for i, example in enumerate(all_scores[-3:]):
        print(f"\nExample {example['id']}:")
        print(f"ROUGE-L Score: {example['rougeL']:.4f}")
        print(f"Generated ({example['gen_tokens']} tokens): {example['generated']}")
        print(f"Reference ({example['ref_tokens']} tokens): {example['reference']}")

def save_results(eval_results, output_dir="./evaluation_output"):
    """Save evaluation results to files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(f"{output_dir}/rouge_summary.json", 'w') as f:
        json.dump({
            'averages': eval_results['averages'],
            'distribution': {
                'rouge1': {
                    'min': float(min(eval_results['scores']['rouge1'])) if eval_results['scores']['rouge1'] else 0,
                    'max': float(max(eval_results['scores']['rouge1'])) if eval_results['scores']['rouge1'] else 0,
                    'median': float(np.median(eval_results['scores']['rouge1'])) if eval_results['scores']['rouge1'] else 0,
                    'std_dev': float(np.std(eval_results['scores']['rouge1'])) if eval_results['scores']['rouge1'] else 0
                },
                'rouge2': {
                    'min': float(min(eval_results['scores']['rouge2'])) if eval_results['scores']['rouge2'] else 0,
                    'max': float(max(eval_results['scores']['rouge2'])) if eval_results['scores']['rouge2'] else 0,
                    'median': float(np.median(eval_results['scores']['rouge2'])) if eval_results['scores']['rouge2'] else 0,
                    'std_dev': float(np.std(eval_results['scores']['rouge2'])) if eval_results['scores']['rouge2'] else 0
                },
                'rougeL': {
                    'min': float(min(eval_results['scores']['rougeL'])) if eval_results['scores']['rougeL'] else 0,
                    'max': float(max(eval_results['scores']['rougeL'])) if eval_results['scores']['rougeL'] else 0,
                    'median': float(np.median(eval_results['scores']['rougeL'])) if eval_results['scores']['rougeL'] else 0,
                    'std_dev': float(np.std(eval_results['scores']['rougeL'])) if eval_results['scores']['rougeL'] else 0
                }
            }
        }, f, indent=2)
    
    with open(f"{output_dir}/detailed_scores.json", 'w') as f:
        json.dump(eval_results['all_scores'], f, indent=2)
    
    print(f"\nEvaluation results saved to {output_dir}/")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Calculate ROUGE scores for generated summaries')
    parser.add_argument('--input', type=str, default='/home/grads/zbz5399/zyzou8/data/zbz5399/AIH/evaluation_results.json',
                        help='Path to the evaluation results JSON file')
    parser.add_argument('--output', type=str, default='./evaluation_output',
                        help='Directory to save the evaluation results')
    parser.add_argument('--plots', action='store_true',
                        help='Generate plots of the evaluation results')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input}...")
    results = load_results(args.input)
    print(f"Loaded {len(results)} examples")
    
    # Calculate ROUGE scores
    print("Calculating ROUGE scores...")
    eval_results = calculate_rouge_scores(results)
    
    # Print report
    print_report(eval_results)
    
    # Save results
    save_results(eval_results, args.output)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()